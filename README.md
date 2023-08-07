# [<sub><sub><img src="https://raw.githubusercontent.com/Validark/Validark/master/zig-z.svg" alt="Lua" height="40"></sub></sub>](https://ziglang.org/) Accelerated Zig Parser

A high-throughput tokenizer and parser (soon™️) for the Zig programming language.

So far, a tokenizer implementation is provided. The mainline Zig tokenizer uses a deterministic finite state machine. Those are pretty good for some applications, but tokenizing can often employ the use of other techniques for added speed.

# Results

The test bench fully reads in all of the Zig files under the folders in the `src` folder. In my test I installed the Zig compiler, ZLS, and a few other Zig projects in my `src` folder. The test bench iterates over the source bytes from each Zig file (with added sentinels) and calls the tokenization function on each.

To tokenize 3,276 Zig files with 59,307,924 bytes and 1,303,536 newlines, the original tokenizer and my new tokenizer have the following characteristics:

|  | token memory (mebibytes)| run-time (milliseconds) | throughput (gigabytes per second) |throughput (lines of code per second) |
|:-:|:-:|:-:|:-:|:-:|
| original | 43.44MiB | 192ms  |0.31 GB/s | 6.8M loc/s |
| this | **18.36MiB** | **85ms** | **0.70 GB/s** | **15.3M loc/s** |

That's over twice as fast and less than half the memory! For context, I am using a Zen 3 desktop and I would be interested in hearing from arm64 users who could share some performance numbers. If you continue reading you will see I designed this code with the hope it would be fast on arm64 platforms too.

Oddly enough, I think this code is generally more maintainable too, as adding an operator or keyword to the tokenizer is literally just adding another string into the relevant array. All of the assumptions and tricks I use are explicitly checked for in compile-time assertions (`grep` for `comptime assert`), so violating any of those invariants will result in compile errors that tell you why you can't change certain things.

# Designing for high performance

In the delicate balancing act that is performance optimization, you generally want:

  1. The ability to process more than one thing at once
  2. Fewer unpredicable branches
  3. A linear traversal over a smaller amount of contiguous memory


I try to achieve each of these in the following ways:

1. SIMD, i.e. single-instruction, multiple data. Instead of operating on a single element at a time, you can operate on 16, 32, or 64 elements simultaneously. Instead of going character-by-character, we use SIMD to check for the length of identifiers/keywords, the length of quotes, the length of whitespace, and the length of comments or single-line quotes. This allows us to move quicker than one byte at a time. We also use a SIMD technique to validate proper utf8 conformance, ported from [simdjson](https://github.com/simdjson/simdjson) by [travisstaloch](https://github.com/travisstaloch/) for use in [simdjzon](https://github.com/travisstaloch/simdjzon/). Please note that that particular code is licensed under the Apache license, included at the bottom of the `main.zig` file.
    - I do not actually use SIMD to find "character literals" of the form `'a'` because these are generally extremely short and did not actually give much benefit in tests.

    - We can't and don't want to use SIMD for absolutely everything because:
      - Comments can be inside of quotes and quotes can be inside of comments
        - Selecting which bitstring to match in next is (probably?) not that efficient. You'd have to multiply each vector and then OR all the vectors together, get the next position, then repeat. I might try out this approach, but I doubt it will be that practical. I also note when I look at the arm64 output that it takes *much* more vector instructions than on x86_64, and doing everything in SIMD generates several hundred instructions on arm64. It might still be worth it though, especially on x86_64, but I doubt it.
      - Operators are all over the place and doing everything in SIMD would require a lot of work that's not that bad for scalar code to do.

2. Fewer unpredictable branches can be achieved through:
    - Using SIMD. Using a conventional while loop to capture a completely unpredictable number of characters in the aforementioned categories all but guarantees a branch mispredict every time we exit the loop, and possibly multiple throughout the loop if the branch predictor is having a bad day. Using SIMD, we can instead produce a bitstring with 0's marked in the place corresponding to target characters like the matching `"`, shift the bitstring according to our cursor's position, and count the trailing ones (the reason the bits are the inverse of what you might expect is because when we shift the bitstring it will be filled with 0's). In most cases, a single "count trailing one's" operation is all we need to find the position we are supposed to go to next. No need for a totally unpredictable while loop that goes character-by-character!

    - Using perfect hash functions. Specifically, keywords like `var` and `const` are mapped into a 7 bit address space by a perfect hash function. Identifiers can be checked against the list of keywords by applying the perfect hash function to each identifier and doing a table lookup to find what keyword it may match, then doing a single 16-byte vs 16-byte comparison to see if the identifier matches that keyword. The keywords are padded in memory to be 16 bytes and have a `len` stored in the final byte so we can check that the incoming identifier has the same length as the prospective keyword. We also use Phil Bagwell's array-mapped trie compression technique, meaning we have a 128-bit bitmap and find which position to check using the bitmap, thereby enabling us to have a packed buffer that need not have 128 slots. We do a similar trick for operators.
      - One cool thing I can do because of Zig's comptime execution feature is tell Zig that a dummy operator/keyword is needed when we do not have an operator or keyword which hashes to the maximum 7 bit value, i.e. 127 (because I am hashing these to 7 bits of address space). If an operator or keyword is added or removed which hashed to 127, the comptime logic will automatically remove or add the dummy operator/keyword. Very nifty! At the moment, one of the perfect hash schemes needs a dummy element and the other does not. It's nice knowing that if we make a change like changing the hash function or adding/removing an operator or keyword, it will automatically figure out the right thing to do. These kinds of tricks are not good in conventional programming languages. We either have to do this work at start-up time or, even worse, someone bakes all the assumptions into the code and then changing it becomes a game of Jenga, except harder because the pieces are not all in one place. In Zig, we write it once and compile-time execution takes care of the rest.

    - I use a trick where I just allocate the upper-bound amount of memory for tokens per-file, and use the `resize` facility of the allocator to reclaim the space I did not fill. The nice thing about this trick is I can always assume there is sufficient space, which eliminates the need to check that such a thing is safe.

    - I place sentinels at the end of files (and I place a newline at the front) to make the rest of the code simpler. This allows us to safely go back a character at any point if the perfect hash function wants us to grab the last two characters from an identifier with only one character, and allows us to safely go past the end of the source file as well. By placing `"` and `'` characters at the end of our buffer, we can eliminate bounds-checking in the code that searches for those characters, and simply check whether we hit the sentinel node after the hot loop finishes. We currently don't break out of these for newlines though, which we should probably do. All other validation for these should occur when actually trying to allocate the string or character they are supposed to represent.

    - Some things we do unconditionally that could be hidden behind a branch, but are very inexpensive so there is no point. Other things we hide behind a branch when it's expensive and generally predictable. E.g. utf8 validation is typically just making sure all bytes are less than 128, i.e. 0x80. Once we see some non-ascii sequences, then we have to do the more computationally expensive work of making sure the byte sequence is valid.

3. We reduce memory consumption by not storing start indices explicitly, which typically need to match the address space of the source length. In the case of Zig, where source files are constrained to be at most ~4GiB, only 32 bits of address space is needed for any given file. Thus the goal becomes reducing 32-bit start indices to something smaller. Quasi-succinct schemes for reducing the space consumption of monotonically increasing integer sequences immediately spring to mind, such as [Elias-Fano encoding](https://www.antoniomallia.it/sorted-integers-compression-with-elias-fano-encoding.html). However, we can achieve good space compression by simply storing the length of each token rather than the start index. Because tokens almost always have a length that can fit in a byte, we try to store all lengths in a byte. In the event that the length is too large to be stored in a byte, we store a `0` instead and make the next 4 bytes the true length. This works because tokens cannot have a length of 0, else they would not exist, therefore we can use lengths of `0` to trigger special behavior. We also know that this idea does not affect the upper bound on the number of Token elements we need to allocate because in order for a token to take up 3 times as much space as a typical token, it needs to have a length of at least 256, which the astute reader may note is significantly larger than 3.

# Still to-do

Aside from the to-do's listed in the `main.zig` file, the plan with this is to rewrite the Zig parser which produces the Abstract Syntax Tree as well. I have a number of ideas on how to dramatically improve efficiency there as well. Stay tuned!

My ultimate goal is that this repository will be integrated with the Zig compiler.

# How to use

```
git clone https://github.com/Validark/Accelerated-Zig-Parser.git
```

Next, install one or more Zig projects under the `src` folder.

```
cd Zig-Parser-Experiment/src
git clone https://github.com/ziglang/zig.git
git clone https://github.com/zigtools/zls.git
```

Then run it!

```
cd ..
zig build -Doptimize=ReleaseFast run
```

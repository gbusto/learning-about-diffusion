# Overview

The goal of this to learn diffusion model architecture one step at a time.

Instead of trying to read an entire paper and understand the math, I plan to build a diffusion model from ground up starting with the dumbest, simplest, most barebones parts. No optimizations or anything fancy. Just the raw stuff. Then I can layer in optimizations and things to learn *how* and *why* they help. Sort of how Neural Networks from Scratch is written. But this won't be anywhere near that complete.

I hope to turn this into a mini educational series in case it helps others or even resonates with how other people like to learn; people like me who aren't math-heavy, but are very comfortable coding. So many video and written explanations online dive so deeply into the math of it, or their code is really poorly written and uses vague variable names or diffusion slang/lingo that doesn't make sense to n00bs.

A big part of the reason I'm doing this is because I've been working on [Blocksmith](https://blocksmithai.com) which is a webapp that generates and textures 3D block models from a prompt and/or image, and I need to improve the texturing engine. I've squeezed about as much juice as I can from existing technology, which in this case is [mv-adapter](https://github.com/huanngzh/MV-Adapter) + SDXL. The difference between SDXL and SD3.5 is MASSIVE, and it'd be amazing to have mv-adapter work with the new SD3.5 architecture. In order to be able to do this kind of work though, I want to understand how mv-adapter works. And to understand how it works, I need to see how it plugs into SDXL's architecture. And in order for that to make sense, it would help to understand how diffusion models work at their core.

I could probably yeet this kind of project and brute force it with AI and me acting as a PM, but then I'd miss a good learning opportunity. So I'm taking this time to learn and share.
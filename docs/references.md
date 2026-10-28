# References

Materials I've been using before and during the hackathon.

## Background: Before knowing the hackathon

Before getting to know the Emmy Hackathon 2026 through LinkedIn, I was self-studying about Parallel Proramming for about a month. My starter resources was:
1. [Stanford CS149: PARALLEL COMPUTING](https://gfxcourses.stanford.edu/cs149/fall25/)
2. [Programming Massively Parallel Processors: A Hands-on Approach by by Wen-mei W. Hwu (Author), David B. Kirk (Author), Izzat El Hajj (Author)](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands-dp-0443439001/dp/0443439001)

I have finished a few first lectures and chapters of those 2 materials above before participating the Hackathon.

## When the Hackathon about to kick off

When preparing before the Hackathon go live, I watched Mr. Dmitry's lecture on Building a GPU Compiler from Scratch on Youtube [Mr. Dmitry's Lecture](https://www.youtube.com/watch?v=EKf0cPYq4e4).
After finished watching the lecture, I found the research blog from Mr. Dmitry so I can understand more clearly:
1. [A Principled ML Compiler Stack in 5,000 Lines of Python — Part 1](https://riftstack.ai/research/building-gpu-compiler-from-scratch-1)
2. [A Principled ML Compiler Stack in 5,000 Lines of Python — Part 2](https://riftstack.ai/research/building-gpu-compiler-from-scratch-2)

The lecture with the blogs above have helped me a lot to have some initial, solid abstract understanding about the ML Compiler stack with important terminologies need to get familiar with.

I also joined the Office Hours held by Mr. Dmitry, Mr. Yashasvi and Mr. Heiko to get to know more about the competition. And I was provided with useful information from the everyone, especially the valuable advice from Mr. Dmitry about recommendedd materials to begin in Compiler domain.


## While building

While building for the Hackathon, my targets are:
- Deeply understand about GPU architecture: I focus on the Programming Massively Parallel Processors book, especially on the fundamental concepts about Data Structure, Compute Architecture, and Memory Architecture.
- Getting familiar with PTX Programming Language: I focus on [PTX Documentations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) so I could grab solid general understanding about PTX programs.  


So my strategy while building for the hackathon is this loop:
1. Spend some time to gain solid foundations about the task need to be done, stick to [TASK.md](TASK.md) and using the main resources as mentioned above.
2. Dive deeper into the task, the [examples](examples), the [problems](problems) to clearly understand them.
3. Whenever I got stuck or not understand how the Compiler work for the task yet, I get back to the book and documentation, with Cursor help, till fully understand what everything work, how they work, and why work that way.

I go one category at a time, and the compiler from the previous category is the foundation. I only add the lowering and the ops that the category I am on needs, from what its own problems show. When it passes, I submit the earlier categories again, to check the new compiler still covers them, then I move on.

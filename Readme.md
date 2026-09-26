This folder contains the code for the paper "Auditing of Unlearning Algorithms". Each subfolder has its own README with setup and run instructions.

- `certified_unlearning_plus_grad_based_methods/`: CIFAR-100 code for model clipping (certified noisy fine-tuning) and gradient-based unlearning methods (e.g. ascent–descent). It also generates the CIFAR-100 data splits used by `r2d_audit/`.
- `hessian-unlearning/`: CIFAR-100 code for the Hessian-based unlearning method.
- `r2d_audit/`: CIFAR-100 code for the Rewind-to-Delete (R2D) algorithm.
- `class_unlearning_audits/`: CIFAR-100 audits of class-centric unlearning (DELETE, Bad Teacher, SCRUB, SCRUB+R).
- `shakeshpere_plays_audit/`: all the Shakespeare runs (character-level LSTM).
- `llama_unlearning/`: LLM-scale audit on TOFU with Llama-3.2-1B-Instruct.
- `convex_unlearning_output_perturbation/`: the output perturbation and Newton-based convex unlearning algorithms.

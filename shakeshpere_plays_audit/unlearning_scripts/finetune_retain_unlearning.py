"""
Finetune-Retain unlearning integration for Shakespeare plays base
"""

from engine import evaluate
from trainer_utils import build_optimizer, build_scheduler
import torch
import time

def unlearn_finetune_retain(model, retain_loader, val_loader, train_loader, forget_loader, criterion, device, run_dir, logger, unlearning_cfg):
	"""Phase 2: Finetune on retain only (finetune_retain strategy)."""
	logger.info("Phase 2: Finetuning on retain only...")

	# Get phase 2 config from unlearning config
	ft_cfg = unlearning_cfg["finetune_retain"]
	weight_decay = ft_cfg["weight_decay"]
	clip = unlearning_cfg["clip"]
	optimizer_name = unlearning_cfg["optimizer"]

	cfg_finetune = {
		"epochs": ft_cfg["phase_2_epochs"],
		"batch_size": unlearning_cfg["batch_size"],
		"lr": ft_cfg["phase_2_lr"],
		"clip": clip,
		"optimizer": optimizer_name,
		"weight_decay": weight_decay,
	}

	logger.info("Phase 2 Config: epochs={}, lr={}, weight_decay={}".format(
		cfg_finetune['epochs'], cfg_finetune['lr'], cfg_finetune['weight_decay']))

	optimizer = build_optimizer(model, cfg_finetune)
	scheduler = build_scheduler(optimizer, cfg_finetune, num_epochs=cfg_finetune["epochs"])
	best_eval_acc_p2 = 0.0
	start_time = time.time()

	for epoch in range(1, cfg_finetune["epochs"] + 1):
		model.train()
		train_loss, train_acc, num_batches = 0.0, 0.0, 0

		for batch_x, batch_y in retain_loader:
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)
			optimizer.zero_grad()
			output = model(batch_x)
			# Handle tuple output from model
			logits = output[0] if isinstance(output, tuple) else output
			loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
			if cfg_finetune['weight_decay'] > 0:
				l2_penalty = 0.5 * sum(p.pow(2).sum() for p in model.parameters())
				loss = loss + cfg_finetune['weight_decay'] * l2_penalty
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_finetune.get("clip", 1.0))
			optimizer.step()
			train_loss += loss.item()
			acc = (logits.argmax(-1) == batch_y).float().mean().item()
			train_acc += acc
			num_batches += 1

		train_loss /= num_batches
		train_acc /= num_batches

		model.eval()
		val_loss, val_acc, val_batches = 0.0, 0.0, 0
		with torch.no_grad():
			for batch_x, batch_y in val_loader:
				batch_x = batch_x.to(device)
				batch_y = batch_y.to(device)
				output = model(batch_x)
				logits = output[0] if isinstance(output, tuple) else output
				loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
				val_loss += loss.item()
				acc = (logits.argmax(-1) == batch_y).float().mean().item()
				val_acc += acc
				val_batches += 1

		val_loss /= val_batches
		val_acc /= val_batches

		best_eval_acc_p2 = max(best_eval_acc_p2, val_acc)
		torch.save(model.state_dict(), str(run_dir / "model_unlearnt.pt"))
		current_lr = optimizer.param_groups[0]['lr']
		logger.info("Epoch {:3d} | LR: {:.6f} | TrLoss: {:.4f} | TrAcc: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} -> SAVED_LATEST".format(
			epoch, current_lr, train_loss, train_acc, val_loss, val_acc))

		if scheduler is not None:
			scheduler.step()

	phase2_time = time.time() - start_time
	logger.info("\nPhase 2 completed in {:.1f}s, best_val_acc={:.4f}".format(phase2_time, best_eval_acc_p2))

	model.load_state_dict(torch.load(str(run_dir / "model_unlearnt.pt"), map_location=device))

	model.eval()
	forget_loss_p2, forget_acc_p2, forget_ppl_p2 = evaluate(model, forget_loader, criterion, device)
	retain_loss_p2, retain_acc_p2, retain_ppl_p2 = evaluate(model, retain_loader, criterion, device)
	combined_loss_p2, combined_acc_p2, combined_ppl_p2 = evaluate(model, train_loader, criterion, device)

	logger.info("\nPhase 2 - Val Metrics on Data Splits:")
	logger.info("  Forget: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(forget_loss_p2, forget_acc_p2, forget_ppl_p2))
	logger.info("  Retain: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(retain_loss_p2, retain_acc_p2, retain_ppl_p2))
	logger.info("  Combined Train: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(combined_loss_p2, combined_acc_p2, combined_ppl_p2))

	metrics = {
		"strategy": "finetune_retain",
		"phase_2_best_val_acc": best_eval_acc_p2,
		"phase_2_duration": phase2_time,
	}
	return model, metrics

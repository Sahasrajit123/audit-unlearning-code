"""
Ascent-Descent unlearning integration for Shakespeare plays base
"""

from engine import evaluate
from trainer_utils import build_optimizer, build_scheduler
import torch
import time

def unlearn_ascent_descent(model, forget_loader, retain_loader, train_loader, val_loader, criterion, device, run_dir, logger, unlearning_cfg=None):
	logger.info("="*80)
	logger.info("Strategy: ascent_descent - Starting unlearning...")
	logger.info("="*80)

	if unlearning_cfg is None:
		unlearning_cfg = {}

	torch.save(model.state_dict(), str(run_dir / "model_before_unlearning.pt"))

	model.eval()
	forget_loss_p1, forget_acc_p1, forget_ppl_p1 = evaluate(model, forget_loader, criterion, device)
	retain_loss_p1, retain_acc_p1, retain_ppl_p1 = evaluate(model, retain_loader, criterion, device)
	combined_loss_p1, combined_acc_p1, combined_ppl_p1 = evaluate(model, train_loader, criterion, device)

	logger.info("\nPhase 1 (Before Unlearning) - Val Metrics on Data Splits:")
	logger.info("  Forget: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(forget_loss_p1, forget_acc_p1, forget_ppl_p1))
	logger.info("  Retain: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(retain_loss_p1, retain_acc_p1, retain_ppl_p1))
	logger.info("  Combined Train: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(combined_loss_p1, combined_acc_p1, combined_ppl_p1))

	ad_config = unlearning_cfg["ascent_descent"]
	clip = unlearning_cfg["clip"]
	optimizer_name = unlearning_cfg["optimizer"]
	batch_size = unlearning_cfg["batch_size"]

	num_epochs = ad_config["epochs"]
	q = ad_config.get("q")
	lambda_coef = ad_config.get("lambda_coef", 0.5)
	forget_epochs_ratio = ad_config["forget_epochs_ratio"]
	forget_epochs = max(1, int(forget_epochs_ratio * num_epochs))
	weight_decay = ad_config["weight_decay"]

	# Parse q (may be None for pure two-phase mode, numeric, or fraction string)
	pure_two_phase = q is None
	if not pure_two_phase:
		q = float(q)

	def _build_cfg(lr_key, n_epochs):
		cfg = {
			"epochs": n_epochs,
			"batch_size": batch_size,
			"lr": ad_config[lr_key],
			"clip": clip,
			"optimizer": optimizer_name,
		}
		for key in ("scheduler", "scheduler_step_size", "scheduler_gamma"):
			if key in ad_config:
				cfg[key] = ad_config[key]
		return cfg

	def _eval_val(model):
		val_loss, val_acc, val_batches = 0.0, 0.0, 0
		with torch.no_grad():
			for bx, by in val_loader:
				bx, by = bx.to(device), by.to(device)
				output = model(bx)
				logits = output[0] if isinstance(output, tuple) else output
				val_loss += criterion(logits.view(-1, logits.size(-1)), by.view(-1)).item()
				val_acc += (logits.argmax(-1) == by).float().mean().item()
				val_batches += 1
		return val_loss / val_batches, val_acc / val_batches

	def _retain_descent_epoch(model, optimizer, epoch_label):
		"""Pure descent on retain set — shared between both modes."""
		model.train()
		epoch_loss, steps = 0.0, 0
		for batch_x, batch_y in retain_loader:
			batch_x, batch_y = batch_x.to(device), batch_y.to(device)
			optimizer.zero_grad()
			output = model(batch_x)
			logits = output[0] if isinstance(output, tuple) else output
			loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
			if weight_decay > 0:
				loss = loss + weight_decay * 0.5 * sum(p.pow(2).sum() for p in model.parameters())
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
			optimizer.step()
			epoch_loss += loss.item()
			steps += 1
		model.eval()
		val_loss, val_acc = _eval_val(model)
		current_lr = optimizer.param_groups[0]['lr']
		avg_loss = epoch_loss / steps if steps else 0
		logger.info("{} | LR: {:.6f} | Loss: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} -> SAVED_LATEST".format(
			epoch_label, current_lr, avg_loss, val_loss, val_acc))
		return val_loss

	start_time = time.time()
	best_val_loss = float("inf")

	# ------------------------------------------------------------------ #
	# Pure two-phase mode: q=null -> ascent-only then descent-only        #
	# ------------------------------------------------------------------ #
	if pure_two_phase:
		ascent_lr = ad_config.get("ascent_lr", ad_config["lr"])
		retain_epochs = num_epochs - forget_epochs

		logger.info("\nPure Ascent-Descent Config (q=None):")
		logger.info("  total_epochs={}, ascent_epochs={}, descent_epochs={}, ascent_lr={}, descent_lr={}, weight_decay={}".format(
			num_epochs, forget_epochs, retain_epochs, ascent_lr, ad_config["lr"], weight_decay))

		cfg_ascent = {"epochs": forget_epochs, "batch_size": batch_size, "lr": ascent_lr, "clip": clip, "optimizer": optimizer_name}
		for key in ("scheduler", "scheduler_step_size", "scheduler_gamma"):
			if key in ad_config:
				cfg_ascent[key] = ad_config[key]

		# Phase 1: gradient ascent on forget set
		logger.info("\n--- Phase 1: Gradient Ascent on Forget Set ({} epochs) ---".format(forget_epochs))
		optimizer = build_optimizer(model, cfg_ascent)
		scheduler = build_scheduler(optimizer, cfg_ascent, num_epochs=forget_epochs)

		for epoch in range(1, forget_epochs + 1):
			model.train()
			epoch_loss, steps = 0.0, 0
			for batch_x, batch_y in forget_loader:
				batch_x, batch_y = batch_x.to(device), batch_y.to(device)
				optimizer.zero_grad()
				output = model(batch_x)
				logits = output[0] if isinstance(output, tuple) else output
				loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
				if weight_decay > 0:
					loss = loss + weight_decay * 0.5 * sum(p.pow(2).sum() for p in model.parameters())
				(-loss).backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
				optimizer.step()
				epoch_loss += loss.item()
				steps += 1
			model.eval()
			val_loss, val_acc = _eval_val(model)
			best_val_loss = min(best_val_loss, val_loss)
			torch.save(model.state_dict(), str(run_dir / "model_unlearnt.pt"))
			current_lr = optimizer.param_groups[0]['lr']
			logger.info("Ascent Epoch {:3d} | LR: {:.6f} | ForgetLoss: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} -> SAVED_LATEST".format(
				epoch, current_lr, epoch_loss / steps if steps else 0, val_loss, val_acc))
			if scheduler is not None:
				scheduler.step()

		torch.save(model.state_dict(), str(run_dir / "model_unlearnt_after_forget_phase.pt"))
		logger.info("Saved checkpoint after ascent phase.")

		model.eval()
		fl, fa, fp = evaluate(model, forget_loader, criterion, device)
		rl, ra, rp = evaluate(model, retain_loader, criterion, device)
		cl, ca, cp = evaluate(model, train_loader, criterion, device)
		vl, va, vp = evaluate(model, val_loader, criterion, device)
		logger.info("\nAfter Ascent Phase - Metrics:")
		logger.info("  Forget: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(fl, fa, fp))
		logger.info("  Retain: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(rl, ra, rp))
		logger.info("  Combined Train: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(cl, ca, cp))
		logger.info("  Val/Test: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(vl, va, vp))

		# Phase 2: gradient descent on retain set — same loop as standard mode
		logger.info("\n--- Phase 2: Gradient Descent on Retain Set ({} epochs) ---".format(retain_epochs))
		if retain_epochs > 0:
			cfg_descent = _build_cfg("lr", retain_epochs)
			optimizer = build_optimizer(model, cfg_descent)
			scheduler = build_scheduler(optimizer, cfg_descent, num_epochs=retain_epochs)
			for epoch in range(1, retain_epochs + 1):
				val_loss = _retain_descent_epoch(model, optimizer, "Descent Epoch {:3d}".format(epoch))
				best_val_loss = min(best_val_loss, val_loss)
				torch.save(model.state_dict(), str(run_dir / "model_unlearnt.pt"))
				if scheduler is not None:
					scheduler.step()

		elapsed = time.time() - start_time
		logger.info("\nPure ascent-descent completed in {:.1f}s".format(elapsed))

		model.load_state_dict(torch.load(str(run_dir / "model_unlearnt.pt"), map_location=device))
		model.eval()
		fl, fa, fp = evaluate(model, forget_loader, criterion, device)
		rl, ra, rp = evaluate(model, retain_loader, criterion, device)
		cl, ca, cp = evaluate(model, train_loader, criterion, device)
		vl, va, vp = evaluate(model, val_loader, criterion, device)
		logger.info("\nAfter Descent Phase - Final Metrics:")
		logger.info("  Forget: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(fl, fa, fp))
		logger.info("  Retain: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(rl, ra, rp))
		logger.info("  Combined Train: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(cl, ca, cp))
		logger.info("  Val/Test: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(vl, va, vp))

		return model, {
			"strategy": "ascent_descent",
			"mode": "pure_two_phase",
			"epochs": num_epochs,
			"forget_epochs": forget_epochs,
			"retain_epochs": retain_epochs,
			"ascent_lr": ascent_lr,
			"descent_lr": ad_config["lr"],
			"best_val_loss": best_val_loss,
			"elapsed_seconds": elapsed,
		}

	# ------------------------------------------------------------------ #
	# Standard interleaved mode: q is a numeric ratio                     #
	# ------------------------------------------------------------------ #
	cfg_unlearn = _build_cfg("lr", num_epochs)

	logger.info("\nAscent-Descent Config:")
	logger.info("  epochs={}, lr={}, forget_epochs={}, q={}, lambda={}, weight_decay={}".format(
		num_epochs, cfg_unlearn["lr"], forget_epochs, q, lambda_coef, weight_decay))

	optimizer = build_optimizer(model, cfg_unlearn)

	steps_per_epoch = len(retain_loader)
	forget_steps = forget_epochs * steps_per_epoch

	scheduler = build_scheduler(optimizer, cfg_unlearn, num_epochs=num_epochs)

	forget_iter = iter(forget_loader)

	def should_do_combined_step(combined_count, retain_count, q):
		if q >= 1:
			return retain_count >= q * (combined_count + 1)
		else:
			return combined_count < (retain_count + 1) / q

	for epoch in range(1, num_epochs + 1):
		model.train()
		epoch_loss = 0.0
		combined_steps = 0
		retain_steps = 0

		for step, (batch_x, batch_y) in enumerate(retain_loader):
			global_step = (epoch - 1) * steps_per_epoch + step
			in_forget_phase = global_step < forget_steps

			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)

			if in_forget_phase and should_do_combined_step(combined_steps, retain_steps, q):
				try:
					forget_batch_x, forget_batch_y = next(forget_iter)
				except StopIteration:
					forget_iter = iter(forget_loader)
					forget_batch_x, forget_batch_y = next(forget_iter)

				forget_batch_x = forget_batch_x.to(device)
				forget_batch_y = forget_batch_y.to(device)

				optimizer.zero_grad()
				output = model(batch_x)
				logits = output[0] if isinstance(output, tuple) else output
				loss_retain = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
				if weight_decay > 0:
					l2_penalty = 0.5 * sum(p.pow(2).sum() for p in model.parameters())
					loss_retain = loss_retain + weight_decay * l2_penalty
				loss_retain.backward()
				retain_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

				optimizer.zero_grad()
				output_f = model(forget_batch_x)
				logits_f = output_f[0] if isinstance(output_f, tuple) else output_f
				loss_forget = criterion(logits_f.view(-1, logits_f.size(-1)), forget_batch_y.view(-1))
				if weight_decay > 0:
					l2_penalty = 0.5 * sum(p.pow(2).sum() for p in model.parameters())
					loss_forget = loss_forget + weight_decay * l2_penalty
				loss_forget.backward()
				forget_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

				optimizer.zero_grad()
				for n, p in model.named_parameters():
					g_r = retain_grads.get(n, torch.zeros_like(p.data))
					g_f = forget_grads.get(n, torch.zeros_like(p.data))
					p.grad = g_r - lambda_coef * g_f

				torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_unlearn.get("clip", 1.0))
				optimizer.step()

				epoch_loss += (loss_retain.item() + loss_forget.item())
				combined_steps += 1
			else:
				optimizer.zero_grad()
				output = model(batch_x)
				logits = output[0] if isinstance(output, tuple) else output
				loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
				if weight_decay > 0:
					l2_penalty = 0.5 * sum(p.pow(2).sum() for p in model.parameters())
					loss = loss + weight_decay * l2_penalty
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_unlearn.get("clip", 1.0))
				optimizer.step()
				epoch_loss += loss.item()
				retain_steps += 1

		model.eval()
		val_loss, val_acc = _eval_val(model)
		avg_loss = epoch_loss / (retain_steps + combined_steps) if (retain_steps + combined_steps) > 0 else 0

		best_val_loss = min(best_val_loss, val_loss)
		torch.save(model.state_dict(), str(run_dir / "model_unlearnt.pt"))

		if epoch == forget_epochs:
			torch.save(model.state_dict(), str(run_dir / "model_unlearnt_after_forget_phase.pt"))
			current_lr = optimizer.param_groups[0]['lr']
			logger.info("Epoch {:3d} | LR: {:.6f} | Loss: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} | Combined: {}, Retain: {} -> SAVED_LATEST + SAVED_FORGET_PHASE".format(
				epoch, current_lr, avg_loss, val_loss, val_acc, combined_steps, retain_steps))
		else:
			current_lr = optimizer.param_groups[0]['lr']
			logger.info("Epoch {:3d} | LR: {:.6f} | Loss: {:.4f} | VLoss: {:.4f} | VAcc: {:.4f} | Combined: {}, Retain: {} -> SAVED_LATEST".format(
				epoch, current_lr, avg_loss, val_loss, val_acc, combined_steps, retain_steps))

		if scheduler is not None:
			scheduler.step()

	model.load_state_dict(torch.load(str(run_dir / "model_unlearnt_after_forget_phase.pt"), map_location=device))
	elapsed = time.time() - start_time
	logger.info("\nAscent-descent completed in {:.1f}s".format(elapsed))

	model.eval()
	forget_loss_p1, forget_acc_p1, forget_ppl_p1 = evaluate(model, forget_loader, criterion, device)
	retain_loss_p1, retain_acc_p1, retain_ppl_p1 = evaluate(model, retain_loader, criterion, device)
	combined_loss_p1, combined_acc_p1, combined_ppl_p1 = evaluate(model, train_loader, criterion, device)
	val_loss_p1, val_acc_p1, val_ppl_p1 = evaluate(model, val_loader, criterion, device)

	logger.info("\nPhase 1 (After Forget Phase) - Val Metrics on Data Splits:")
	logger.info("  Forget: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(forget_loss_p1, forget_acc_p1, forget_ppl_p1))
	logger.info("  Retain: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(retain_loss_p1, retain_acc_p1, retain_ppl_p1))
	logger.info("  Combined Train: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(combined_loss_p1, combined_acc_p1, combined_ppl_p1))
	logger.info("  Val/Test: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(val_loss_p1, val_acc_p1, val_ppl_p1))

	model.load_state_dict(torch.load(str(run_dir / "model_unlearnt.pt"), map_location=device))
	model.eval()
	forget_loss, forget_acc, forget_ppl = evaluate(model, forget_loader, criterion, device)
	retain_loss, retain_acc, retain_ppl = evaluate(model, retain_loader, criterion, device)
	combined_loss, combined_acc, combined_ppl = evaluate(model, train_loader, criterion, device)
	val_loss, val_acc, val_ppl = evaluate(model, val_loader, criterion, device)

	logger.info("\nPhase 2 (After Retain Phase) - Val Metrics on Data Splits:")
	logger.info("  Forget: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(forget_loss, forget_acc, forget_ppl))
	logger.info("  Retain: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(retain_loss, retain_acc, retain_ppl))
	logger.info("  Combined Train: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(combined_loss, combined_acc, combined_ppl))
	logger.info("  Val/Test: loss={:.4f}, acc={:.4f}, ppl={:.2f}".format(val_loss, val_acc, val_ppl))

	metrics = {
		"strategy": "ascent_descent",
		"epochs": num_epochs,
		"forget_epochs": forget_epochs,
		"q": q,
		"lambda_coef": lambda_coef,
		"best_val_loss": best_val_loss,
		"elapsed_seconds": elapsed,
	}

	return model, metrics

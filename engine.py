from tqdm import tqdm


def run_train(
    model, train_dataloader, optimizer, scheduler, seg_loss_func, metrics, accelerate
):  # , pseudo_df, trn_idxs_list, val_idxs_list):
    model.train()

    progress_bar = tqdm(train_dataloader, total=len(train_dataloader))

    for images, masks in progress_bar:
        optimizer.zero_grad()
        prediction = model.forward(images)[0]
        loss = seg_loss_func(prediction, masks)
        accelerate.backward(loss)

        # loss.backward()
        optimizer.step()
        scheduler.step()

        iou_score = metrics(prediction, masks)
        progress_bar.set_description(
            f"IoU: {iou_score:.4f} loss: {loss.item():.4f} lr: {optimizer.param_groups[0]['lr']:.6f}"
        )
    return {
        "loss": loss.item(),
        "iou": iou_score,
        "images": images,
        "masks": masks,
        "predictions": prediction
    }
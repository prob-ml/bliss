import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def isnan(x):
    return x != x

def get_one_hot_encoding_from_int(z, n_classes):
    z_one_hot = torch.zeros(len(z), n_classes).to(device)
    z_one_hot.scatter_(1, z.view(-1, 1), 1)
    z_one_hot = z_one_hot.view(len(z), n_classes)

    return z_one_hot

def get_categorical_loss(log_probs, true_n_stars):
    assert torch.all(log_probs <= 0)
    assert log_probs.shape[0] == len(true_n_stars)
    max_detections = log_probs.shape[1]

    return torch.sum(
        -log_probs * \
        get_one_hot_encoding_from_int(true_n_stars.long(),
                                        max_detections), dim = 1)

def eval_star_counter_loss(star_counter, train_loader,
                            optimizer = None, train = True):

    avg_loss = 0.0
    max_detections = torch.Tensor([star_counter.max_detections])

    for _, data in enumerate(train_loader):
        images = data['image'].to(device)
        true_n_stars = data['n_stars'].to(device)

        if train:
            star_counter.train()
            assert optimizer is not None
            optimizer.zero_grad()
        else:
            star_counter.eval()

        # evaluate log q
        log_probs = star_counter(images)
        loss = get_categorical_loss(log_probs, true_n_stars).mean()

        assert not isnan(loss)

        if train:
            loss.backward()
            optimizer.step()

        avg_loss += loss * images.shape[0] / len(train_loader.dataset)

    return avg_loss

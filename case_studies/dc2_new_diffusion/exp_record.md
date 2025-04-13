## DC2 New Diffusion Experiment Record

### 01-24

7. 

```python
self.init_conv = nn.Sequential(
            ConvBlock(self.in_channel, self.dim, kernel_size=5),
            *[ConvBlock(self.dim, self.dim, kernel_size=5) for _ in range(2)],
            C3(self.dim, self.dim, n=3),
        )

self.final_conv = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=3),
            C3(self.dim, self.dim, n=3),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.out_channel, kernel_size=(1, 1))
        )
```

8. 

```python
self.init_conv = nn.Sequential(
            ConvBlock(self.in_channel, self.dim, kernel_size=5),
            *[ConvBlock(self.dim, self.dim, kernel_size=5) for _ in range(3)],
            C3(self.dim, self.dim, n=3),
        )

self.final_conv = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=3),
            C3(self.dim, self.dim, n=3),
            ConvBlock(self.dim, self.dim, kernel_size=3),
            C3(self.dim, self.dim, n=3),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.out_channel, kernel_size=(1, 1))
        )
```

9. 

```python
self.unet = UNet(in_channel=target_ch + len(self.survey_bands),
                         out_channel=target_ch,
                         dim=64,
                         attn_heads=8)

self.init_conv = nn.Sequential(
            ConvBlock(self.in_channel, self.dim, kernel_size=5),
            *[ConvBlock(self.dim, self.dim, kernel_size=5) for _ in range(3)],
            C3(self.dim, self.dim, n=3),
        )

self.final_conv = nn.Sequential(
            # ConvBlock(self.dim, self.dim, kernel_size=3),
            # C3(self.dim, self.dim, n=3),
            ConvBlock(self.dim, self.dim, kernel_size=3),
            C3(self.dim, self.dim, n=3),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.out_channel, kernel_size=(1, 1))
        )
```

10. 

```python
self.unet = UNet(in_channel=target_ch + len(self.survey_bands),
                         out_channel=target_ch,
                         dim=64,
                         attn_heads=4,
                         attn_head_dim=64)

self.init_conv = nn.Sequential(
            ConvBlock(self.in_channel, self.dim, kernel_size=5),
            *[ConvBlock(self.dim, self.dim, kernel_size=5) for _ in range(3)],
            C3(self.dim, self.dim, n=3),
        )

self.final_conv = nn.Sequential(
            # ConvBlock(self.dim, self.dim, kernel_size=3),
            # C3(self.dim, self.dim, n=3),
            ConvBlock(self.dim, self.dim, kernel_size=3),
            C3(self.dim, self.dim, n=3),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, self.out_channel, kernel_size=(1, 1))
        )
```


### 03-16 (ynet_full_diffusion)

1. assertion error because n_sources exceeds 3
2. assertion error because n_sources exceeds 3
3. clamp n_sources
4. clamp flux max to be 22025


### 03-17 (ynet_full_diffusion)

1. clamp flux max to be 22025; use reweighted mean internal loss
2. don't predict flux


### 03-18 (ynet_full_diffusion)

1. use reweighted mean internal loss; set acc_gradient_batches to 10
2. use reweighted mean internal loss; set acc_gradient_batches to 10; let n_sources to be either 0 or 1
3. use reweighted mean internal loss; set acc_gradient_batches to 10; let n_sources to be either 0 or 1; set lr to 1e-2
4. use reweighted mean internal loss; set acc_gradient_batches to 10; let n_sources to be either 0 or 1


### 03-28 (ynet_full_diffusion)

1. clamp flux max to be 22025; use reweighted mean internal loss; set acc_gradient_batches to 10; let n_sources to be either 0 or 1; let null flux to be 1
2. clamp flux max to be 22025; use reweighted mean internal loss; set acc_gradient_batches to 10; let n_sources to be either 0 or 1; let null flux to be -1
3. clamp flux max to be 22025; use reweighted mean internal loss; let n_sources to be either 0 or 1; let null flux to be -1
4. clamp flux max to be 22025; use reweighted mean internal loss; let n_sources to be either 0 or 1; let null flux to be 0

(find that I'm using the old log normalized factor, null flux doesn't take effect)


### 03-30 (simple_net_diffusion)

1. num_pre_cond_layers = 1; num_post_cond_layers = 1;
2. num_pre_cond_layers = 2; num_post_cond_layers = 2;
3. num_pre_cond_layers = 4; num_post_cond_layers = 4;


### 03-31 (ynet_full_diffusion)

1. clamp flux max to be 22025; use reweighted mean internal loss; let n_sources to be either 0 or 1; let null flux to be -1


### 04-04 (simple_net_diffusion)

1. v2 model; 8 layers; w/ spatial
2. v2 model; 8 layers; w/o spatial


### 04-04 (ynet_full_diffusion)

1. v2 model; dim=32
2. v2 model; dim=64

(find that the `encode` function in DiffusionFactor will re-encode null flux to 0)


### 04-07 (ynet_full_diffusion)

1. v2 model; dim=64


### 04-07 (simple_net_diffusion)

1. v2 model; 8 layers; w/o spatial
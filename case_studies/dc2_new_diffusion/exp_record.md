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



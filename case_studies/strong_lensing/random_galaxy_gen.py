import random
import sys

n1 = random.uniform(2.5, 4.4)
half_light_radius = random.uniform(1, 3)
flux1 = random.uniform(0.2, 0.4)

n2 = random.uniform(0.5, 2.5)
scale_radius = random.uniform(0.5, 1.5)
flux2 = random.uniform(0.6, 0.8)
q = random.uniform(0.5, 0.9)

beta = random.randint(0, 90)

size = int(sys.argv[1])

# X-Y coordinates in final image
x = random.randint(0, size)
y = random.randint(0, size)


gal_params = [n1, half_light_radius, flux1, n2, scale_radius, flux2, q, beta, x, y]
print(" ".join(map(str, gal_params)))  # noqa: WPS421

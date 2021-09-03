import rpy2
import numpy as np
from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
numpy2ri.activate()

class TruncMVNorm:
    def __init__(self) -> None:
        self._r_mtmvnorm = robjects.r("""
            function(m, l, u) {
                tmvtnorm::mtmvnorm(
                    as.vector(m), 
                    lower=as.vector(l), 
                    upper=as.vector(u),
                    doComputeVariance=TRUE
                )
            }
        """
        )

        self._r_pmvnorm = robjects.r("""
            function(l, u, m) mvtnorm::pmvnorm(
                as.vector(l), 
                as.vector(u), 
                as.vector(m))
        """)

    def mtmvnorm(self, mu, sigma, lowers, uppers, clip=True):
        numpy2ri.activate()
        assert len(lowers.shape) == 2
        assert len(uppers.shape) == 2
        assert lowers.shape[0] == uppers.shape[0]
        assert lowers.shape[1] == uppers.shape[1]
        assert lowers.shape[1] == mu.shape[0]

        means = []
        vars = []
        for i in range(lowers.shape[0]):
            mu_zero = np.zeros_like(mu)
            lower_zero = lowers[i] - mu
            upper_zero = uppers[i] - mu
            mean, var = self._r_mtmvnorm(mu_zero, lower_zero, upper_zero)
            if np.any(np.isnan(mean)):
                mean = (lower_zero + upper_zero) / 2
            if clip:
                for j in range(lowers.shape[1]):
                    mean[j] = mean[j].clip(lower_zero[j], upper_zero[j])
            #print(f"i={i}, lower={lowers[i]}, upper={uppers[i]}, res={mu + mean}")
            means.append(mu + mean)
            vars.append(var)
        return np.stack(means), np.stack(vars)

    def pmvnorm(self, mu, sigma, lowers, uppers):
        assert len(lowers.shape) == 2
        assert len(uppers.shape) == 2
        assert lowers.shape[0] == uppers.shape[0]
        assert lowers.shape[1] == uppers.shape[1]
        assert lowers.shape[1] == mu.shape[0]

        probs = []
        for i in range(lowers.shape[0]):
            p = self._r_pmvnorm(lowers[i], uppers[i], mu)
            probs.append(p)
        return np.stack(probs)
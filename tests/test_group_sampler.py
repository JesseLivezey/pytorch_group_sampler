"""
Test group sampling functions. Based on scipy's tests.

"""
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_almost_equal, assert_equal,
                           assert_array_less, assert_)
import pytest
from pytest import raises as assert_raises
import numpy as np

import scipy.linalg
from scipy.stats import ks_2samp, kstest
import torch

from group_sampler import special_ortho_group_rvs, ortho_group_rvs

class TestSpecialOrthoGroup(object):
    def test_reproducibility(self):
        torch.manual_seed(514)
        x = special_ortho_group_rvs(3, output_numpy=True)
        torch.manual_seed(514)
        x2 = special_ortho_group_rvs(3, output_numpy=True)
        assert_array_almost_equal(x, x2)

    def test_det_and_ortho(self):
        # Float64
        xs = [special_ortho_group_rvs(dim, output_numpy=True)
              for dim in range(2,12)
              for i in range(3)]

        # Test that determinants are always +1
        dets = [np.linalg.det(x) for x in xs]
        print(dets)
        assert_allclose(dets, [1.]*30, atol=5e-6)

        # Test that these are orthogonal matrices
        for x in xs:
            assert_array_almost_equal(np.dot(x, x.T),
                                      np.eye(x.shape[0]))
        # Float32
        xs = [special_ortho_group_rvs(dim, output_numpy=True)
              for dim in range(2,12)
              for i in range(3)]

        # Test that determinants are always +1
        dets = [np.linalg.det(x) for x in xs]
        assert_allclose(dets, [1.]*30, atol=5e-6)

        # Test that these are orthogonal matrices
        for x in xs:
            assert_array_almost_equal(np.dot(x, x.T),
                                      np.eye(x.shape[0]))

    def test_haar(self):
        # Test that the distribution is constant under rotation
        # Every column should have the same distribution
        # Additionally, the distribution should be invariant under another rotation

        # Generate samples
        dim = 5
        samples = 1000  # Not too many, or the test takes too long
        ks_prob = .05
        torch.manual_seed(515)
        xs = special_ortho_group_rvs(dim, size=samples, output_numpy=True)

        # Dot a few rows (0, 1, 2) with unit vectors (0, 2, 4, 3),
        #   effectively picking off entries in the matrices of xs.
        #   These projections should all have the same disribution,
        #     establishing rotational invariance. We use the two-sided
        #     KS test to confirm this.
        #   We could instead test that angles between random vectors
        #     are uniformly distributed, but the below is sufficient.
        #   It is not feasible to consider all pairs, so pick a few.
        els = ((0,0), (0,2), (1,4), (2,3))
        #proj = {(er, ec): [x[er][ec] for x in xs] for er, ec in els}
        proj = dict(((er, ec), sorted([x[er][ec] for x in xs])) for er, ec in els)
        pairs = [(e0, e1) for e0 in els for e1 in els if e0 > e1]
        ks_tests = [ks_2samp(proj[p0], proj[p1])[1] for (p0, p1) in pairs]
        assert_array_less([ks_prob]*len(pairs), ks_tests)

class TestOrthoGroup(object):
    def test_reproducibility(self):
        torch.manual_seed(516)
        x = ortho_group_rvs(3, output_numpy=True)
        torch.manual_seed(516)
        x2 = ortho_group_rvs(3, output_numpy=True)
        assert_array_almost_equal(x, x2)

    def test_det_and_ortho(self):
        # Float64
        xs = [[ortho_group_rvs(dim, output_numpy=True, dtype=torch.float64)
               for i in range(10)]
              for dim in range(2,12)]

        # Test that abs determinants are always +1
        dets = np.array([[np.linalg.det(x) for x in xx] for xx in xs])
        assert_allclose(np.fabs(dets), np.ones(dets.shape), rtol=1e-13)

        # Test that we get both positive and negative determinants
        # Check that we have at least one and less than 10 negative dets in a sample of 10. The rest are positive by the previous test.
        # Test each dimension separately
        assert_array_less([0]*10, [np.nonzero(d < 0)[0].shape[0] for d in dets])
        assert_array_less([np.nonzero(d < 0)[0].shape[0] for d in dets], [10]*10)

        # Test that these are orthogonal matrices
        for xx in xs:
            for x in xx:
                assert_array_almost_equal(np.dot(x, x.T),
                                          np.eye(x.shape[0]))
        # Float32
        xs = [[ortho_group_rvs(dim, output_numpy=True)
               for i in range(10)]
              for dim in range(2,12)]

        # Test that abs determinants are always +1
        dets = np.array([[np.linalg.det(x) for x in xx] for xx in xs])
        assert_allclose(np.fabs(dets), np.ones(dets.shape), atol=5e-6)

        # Test that we get both positive and negative determinants
        # Check that we have at least one and less than 10 negative dets in a sample of 10. The rest are positive by the previous test.
        # Test each dimension separately
        assert_array_less([0]*10, [np.nonzero(d < 0)[0].shape[0] for d in dets])
        assert_array_less([np.nonzero(d < 0)[0].shape[0] for d in dets], [10]*10)

        # Test that these are orthogonal matrices
        for xx in xs:
            for x in xx:
                assert_array_almost_equal(np.dot(x, x.T),
                                          np.eye(x.shape[0]))

    def test_haar(self):
        # Test that the distribution is constant under rotation
        # Every column should have the same distribution
        # Additionally, the distribution should be invariant under another rotation

        # Generate samples
        dim = 5
        samples = 1000  # Not too many, or the test takes too long
        ks_prob = .05
        torch.manual_seed(517)
        xs = ortho_group_rvs(dim, size=samples, output_numpy=True)

        # Dot a few rows (0, 1, 2) with unit vectors (0, 2, 4, 3),
        #   effectively picking off entries in the matrices of xs.
        #   These projections should all have the same disribution,
        #     establishing rotational invariance. We use the two-sided
        #     KS test to confirm this.
        #   We could instead test that angles between random vectors
        #     are uniformly distributed, but the below is sufficient.
        #   It is not feasible to consider all pairs, so pick a few.
        els = ((0,0), (0,2), (1,4), (2,3))
        #proj = {(er, ec): [x[er][ec] for x in xs] for er, ec in els}
        proj = dict(((er, ec), sorted([x[er][ec] for x in xs])) for er, ec in els)
        pairs = [(e0, e1) for e0 in els for e1 in els if e0 > e1]
        ks_tests = [ks_2samp(proj[p0], proj[p1])[1] for (p0, p1) in pairs]
        assert_array_less([ks_prob]*len(pairs), ks_tests)

    @pytest.mark.slow
    def test_pairwise_distances(self):
        # Test that the distribution of pairwise distances is close to correct.
        np.random.seed(514)

        def random_ortho(dim):
            u, _s, v = np.linalg.svd(np.random.normal(size=(dim, dim)))
            return np.dot(u, v)

        for dim in range(2, 6):
            def generate_test_statistics(rvs, N=1000, eps=1e-10):
                stats = np.array([
                    np.sum((rvs(dim=dim) - rvs(dim=dim))**2)
                    for _ in range(N)
                ])
                # Add a bit of noise to account for numeric accuracy.
                stats += np.random.uniform(-eps, eps, size=stats.shape)
                return stats

            expected = generate_test_statistics(random_ortho)
            actual = generate_test_statistics(scipy.stats.ortho_group.rvs)

            _D, p = scipy.stats.ks_2samp(expected, actual)

            assert_array_less(.05, p)

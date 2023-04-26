

import unittest
from absl.testing import parameterized
from alf.environments.simple.noisy_array import NoisyArray


class NoisyArrayTest(parameterized.TestCase, unittest.TestCase):
    @parameterized.parameters((5, 3), (201, 100))
    def test_noisy_array_environment(self, K, M):
        array = NoisyArray(K, M)
        array.reset()
        for _ in range(K - 1):
            done = array.step(NoisyArray.RIGHT)[2]
        self.assertTrue(done)

        array.reset()
        array.step(NoisyArray.LEFT)  # cannot go beyond the left boundary
        self.assertEqual(array._position, 0)

        array.step(NoisyArray.RIGHT)
        array.reset()

        game_ends = 0
        total_rewards = 0
        done = False
        for _ in range(2 * K - 1):
            if done:
                array.reset()
                done = False
            else:
                _, r, done, _ = array.step(NoisyArray.RIGHT)
                total_rewards += r
                game_ends += int(done)

        self.assertEqual(game_ends, 2)
        self.assertEqual(total_rewards, 2)
        self.assertEqual(r, 1.0)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()

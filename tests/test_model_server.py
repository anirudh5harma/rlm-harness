import unittest

from rlm_harness.model_server import MLXServerConfig


class MLXServerConfigTests(unittest.TestCase):
    def test_builds_mlx_server_command(self):
        config = MLXServerConfig(
            model="mlx-community/test",
            host="0.0.0.0",
            port=9999,
            extra_args=("--foo", "bar"),
        )

        self.assertEqual(config.base_url, "http://0.0.0.0:9999/v1")
        self.assertEqual(
            config.command(),
            [
                "mlx_lm.server",
                "--model",
                "mlx-community/test",
                "--host",
                "0.0.0.0",
                "--port",
                "9999",
                "--foo",
                "bar",
            ],
        )


if __name__ == "__main__":
    unittest.main()

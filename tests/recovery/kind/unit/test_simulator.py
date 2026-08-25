from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from tests.recovery.kind.simulator import SimulatorClient


class SimulatorClientTest(unittest.TestCase):
    @patch("tests.recovery.kind.simulator.urlopen")
    def test_sets_and_clears_xid_over_http(self, urlopen: Mock) -> None:
        response = Mock()
        response.read.return_value = b""
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        urlopen.return_value = response
        client = SimulatorClient("http://127.0.0.1:9400")

        client.set_xid(79)
        set_request = urlopen.call_args_list[0].args[0]
        self.assertEqual("PUT", set_request.get_method())
        self.assertEqual(b'{"xid": 79}', set_request.data)

        client.clear()
        clear_request = urlopen.call_args_list[1].args[0]
        self.assertEqual("DELETE", clear_request.get_method())
        self.assertIsNone(clear_request.data)


if __name__ == "__main__":
    unittest.main()

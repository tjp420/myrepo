"""Stub for core.substrate_interface used in api_server imports."""


def submit_transaction(tx):
    return None


class SubstrateInterface:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, tx):
        return submit_transaction(tx)

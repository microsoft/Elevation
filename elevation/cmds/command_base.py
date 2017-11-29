
class Command(object):

    def execute(self, **kwargs):
        raise NotImplementedError("execute function must be implemented.")

    @classmethod
    def cli_execute(cls):
        cls().execute()

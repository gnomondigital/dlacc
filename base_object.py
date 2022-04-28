class BaseClass:
    def dump_object(self):
        attrs = vars(self)
        content = ""
        if len(attrs):
            content = ", ".join("%s: %s" % item for item in attrs.items())
        return content

    def raise_default_error(self):
        raise RuntimeError(self.dump_object())

    def _print(self, content):
        print("[%s]" % (self.__class__.__name__), end=" ")
        print(content)

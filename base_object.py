class BaseClass:
    def dump_object(self):
        attrs = vars(self)
        # {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
        # now dump this in some way or another
        content = ""
        if len(attrs):
            content = ", ".join("%s: %s" % item for item in attrs.items())
        return content

    def raise_default_error(self):
        raise RuntimeError(self.dump_object())

    def _print(self, content):
        print("[%s]" % (self.__class__.__name__), end=" ")
        print(content)

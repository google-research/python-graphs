import astunparse
import gast as ast


astunparse.Unparser.boolops = {'And': 'and', 'Or': 'or', ast.And: 'and', ast.Or: 'or'}


def _arguments(self, t):
    first = True
    # normal arguments
    all_args = getattr(t, 'posonlyargs', []) + t.args
    defaults = [None] * (len(all_args) - len(t.defaults)) + t.defaults
    for index, elements in enumerate(zip(all_args, defaults), 1):
        a, d = elements
        if first:first = False
        else: self.write(", ")
        self.dispatch(a)
        if d:
            self.write("=")
            self.dispatch(d)
        if index == len(getattr(t, 'posonlyargs', ())):
            self.write(", /")

    # varargs, or bare '*' if no varargs but keyword-only arguments present
    if t.vararg or getattr(t, "kwonlyargs", False):
        if first:first = False
        else: self.write(", ")
        self.write("*")
        if t.vararg:
            if hasattr(t.vararg, 'arg'):
                self.write(t.vararg.arg)
                if t.vararg.annotation:
                    self.write(": ")
                    self.dispatch(t.vararg.annotation)
            else:
                self.write(t.vararg)
                if getattr(t, 'varargannotation', None):
                    self.write(": ")
                    self.dispatch(t.varargannotation)

    # keyword-only arguments
    if getattr(t, "kwonlyargs", False):
        for a, d in zip(t.kwonlyargs, t.kw_defaults):
            if first:first = False
            else: self.write(", ")
            self.dispatch(a),
            if d:
                self.write("=")
                self.dispatch(d)

    # kwargs
    if t.kwarg:
        if first:first = False
        else: self.write(", ")
        if hasattr(t.kwarg, 'arg'):
            self.write("**"+t.kwarg.arg)
            if t.kwarg.annotation:
                self.write(": ")
                self.dispatch(t.kwarg.annotation)
        elif hasattr(t.kwarg, 'id'):  # if this is a gast._arguments
            self.write("**"+t.kwarg.id)
            if t.kwarg.annotation:
                self.write(": ")
                self.dispatch(t.kwarg.annotation)
        else:
            self.write("**"+t.kwarg)
            if getattr(t, 'kwargannotation', None):
                self.write(": ")
                self.dispatch(t.kwargannotation)

astunparse.Unparser._arguments = _arguments

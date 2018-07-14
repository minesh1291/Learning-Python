# Useful Python Functions
https://docs.python.org/3.3/library/functions.html

### Basic

* ```dir([object])```

  Without arguments, return the list of names in the current local scope. With an argument, attempt to return a list of valid attributes for that object.

```Python
    >>> a=2
    >>> b=3
    >>> c=a+b
    >>> dir()
    ['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'a', 'b', 'c']
```

* ```globals()```

  Return a dictionary representing the current global symbol table. This is always the dictionary of the current module (inside a function or method, this is the module where it is defined, not the module from which it is called).

```Python
      >>> globals()
      {'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <class '_frozen_importlib.BuiltinImporter'>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, 'a': 2, 'b': 3, 'c': 5}
```
* ```locals()```

  Update and return a dictionary representing the current local symbol table. Free variables are returned by locals() when it is called in function blocks, but not in class blocks.

  **Note** The contents of this dictionary should not be modified; changes may not affect the values of local and free variables used by the interpreter.

* ```help([object])```

  Invoke the built-in help system. (This function is intended for interactive use.) If no argument is given, the interactive help system starts on the interpreter console. If the argument is a string, then the string is looked up as the name of a module, function, class, method, keyword, or documentation topic, and a help page is printed on the console. If the argument is any other kind of object, a help page on the object is generated.

### Special keywords

* ```yield```
  The yield statement is only used when defining a generator function, and is only used in the body of the generator function. Using a yield statement in a function definition is sufficient to cause that definition to create a generator function instead of a normal function. When a generator function is called, it returns an iterator known as a generator iterator, or more commonly, a generator.

* ```lambda```

# List attributes of an object
>>> class new_class():
...   def __init__(self, number):
...     self.multi = int(number) * 2
...     self.str = str(number)
... 
>>> a = new_class(2)
>>> a.__dict__
{'multi': 4, 'str': '2'}
>>> a.__dict__.keys()
dict_keys(['multi', 'str'])

{{ objname | escape | underline }}

{%- set result = members | reject('in', inherited_members) | list %}

.. autoclass:: {{ fullname }}


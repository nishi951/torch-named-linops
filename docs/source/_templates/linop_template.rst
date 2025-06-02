{{ fullname | escape | underline }}

{%- set result = members | reject('in', inherited_members) | list %}

.. autoclass:: {{ fullname }}

   .. automethod:: __init__

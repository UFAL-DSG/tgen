#!/usr/bin/env python
# coding=utf-8

"""
Helper data structures
"""

import re


class DAI(object):
    """Simple representation of a single dialogue act item."""

    __slots__ = ['da_type', 'slot', 'value']

    def __init__(self, da_type, slot=None, value=None):
        self.da_type = da_type
        self.slot = slot
        self.value = value

    def __unicode__(self):
        if self.slot is None:
            return self.da_type + '()'
        if self.value is None:
            return self.da_type + '(' + self.slot + ')'
        quote = '\'' if (' ' in self.value or ':' in self.value) else ''
        return self.da_type + '(' + self.slot + '=' + quote + self.value + quote + ')'


class DA(object):
    """Dialogue act – basically a list of DAIs."""

    def __init__(self):
        self.dais = []

    def __getitem__(self, idx):
        return self.dais[idx]

    def __setitem__(self, idx, value):
        self.dais[idx] = value

    def append(self, value):
        self.dais.append(value)

    def __unicode__(self):
        return '&'.join([unicode(dai) for dai in self.dais])

    def __str__(self):
        return unicode(self).encode('ascii', errors='xmlcharrefreplace')

    def __len__(self):
        return len(self.dais)

    @staticmethod
    def parse(da_text):
        """Parse a DA string into DAIs (DA types, slots, and values)."""
        da = DA()

        # split acc. to DAIs, trim final bracket
        for dai_text in da_text[:-1].split(')&'):
            da_type, svp = dai_text.split('(', 1)

            if not svp:  # no slot + value (e.g. 'hello()')
                da.append(DAI(da_type))
                continue

            if '=' not in svp:  # no value (e.g. 'request(to_stop)')
                da.append(DAI(da_type, svp))
                continue

            slot, value = svp.split('=', 1)
            if value.startswith('"'):  # remove quotes
                value = value[1:-1]

            da.append(DAI(da_type, slot, value))

        return da

    def value_for_slot(self, slot):
        """Return the value for the given slot (None if unset or not present at all)."""
        for dai in self.dais:
            if dai.slot == slot:
                return dai.value
        return None

    def has_value(self, value):
        """If the DA contains the given value, return the corresponding slot; return None
        otherwise. Abstracts away from "and" and "or" values (returns True for both coordination
        members)."""
        for dai in self.dais:
            if dai.value == value:
                return dai.slot
            if (dai.value is not None and
                    value not in [None, '?'] and
                    (re.match(r'.* (and|or) ' + value + r'$', dai.value) or
                     re.match(r'^' + value + r' (and|or) ', dai.value))):
                return dai.slot
        return None


class Abst(object):
    """Simple representation of a single abstraction/delexicalization instruction."""

    __slots__ = ['slot', 'value', 'surface_form', 'start', 'end']

    def __init__(self, slot=None, value=None, surface_form=None, start=None, end=None):
        self.slot = slot
        self.value = value
        self.surface_form = surface_form
        self.start = start
        self.end = end
        if self.start is not None and self.end is None:
            self.end = self.start + 1

    def __unicode__(self):
        """Create string representation of the abstraction instruction, in the following format:
        slot="value":"surface_form":start-end. Surface form is omitted if None, quotes are omitted
        if not needed."""
        # prepare quoting
        quote_value = '"' if ' ' in self.value or ':' in self.value else ''
        if self.surface_form is not None:
            quote_sf = '"' if ' ' in self.surface_form or ':' in self.surface_form else ''
        # create output
        out = self.slot + '=' + quote_value + self.value + quote_value + ':'
        if self.surface_form is not None:
            out += quote_sf + self.surface_form + quote_sf + ':'
        out += str(self.start) + '-' + str(self.end)
        return out

    @staticmethod
    def parse(abst_str):
        """Create the abstraction instruction from a string representation, in the following
        format: slot="value":"surface_form":start-end. Here, surface form is optional and value
        and surface form do not need to be equoted if they do not contain colons or spaces.
        @param abst_str: string representation of the abstraction instruction
        @return: Abst object representing the abstraction instruction
        """
        slot, rest = abst_str.split('=', 1)
        if rest.startswith('"'):
            value, rest = re.split(r'(?<!\\)":', rest[1:], maxsplit=1)
        else:
            value, rest = rest.split(':', 1)
        if rest.startswith('"'):
            surface_form, rest = re.split(r'(?<!\\)":', rest[1:], maxsplit=1)
        elif ':' in rest:
            surface_form, rest = rest.split(':', 1)
        else:
            surface_form = None
        start, end = [int(part) for part in rest.split('-', 1)]
        return Abst(slot, value, surface_form, start, end)

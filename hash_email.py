#!/usr/bin/env python
# coding: utf-8

from hashlib import sha1


def compute_hash(email):
    return sha1(email.lower().encode('utf-8')).hexdigest()

print(compute_hash('kristiaan@dejongh.de')) # db7651a9d9b6106e0f61310b6bf106bd009e0c54

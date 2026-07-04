"""Typed boundaries between Icepick subsystems.

Allocation, processing, and agent share only what lives here: record shapes,
manifest schemas, action requests, and report envelopes. Concrete subsystems
import from this package and never from each other.
"""

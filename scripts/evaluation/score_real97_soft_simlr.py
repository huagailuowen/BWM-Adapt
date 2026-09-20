#!/usr/bin/env python3
"""Reuse matched tracking/cache rules, without the old initialization comparison."""
import score_real97_soft_balanced9_static_matched as base
import score_real97_soft_family_mean_comparison as reference

if __name__ == '__main__':
    base.process = reference.process
    base.main()

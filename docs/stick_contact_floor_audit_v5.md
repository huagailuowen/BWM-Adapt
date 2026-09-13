# Contact evidence audit v5

This is a paired diagnostic, not a new success classifier. All v4 decisions and
success thresholds remain unchanged. Use only CPU compute nodes.

The specific hypothesis is that flat blue tabletop tape can join a current
body silhouette, especially when the body is only slightly raised. This can
pull the apparent lower edge downward and obscure actual clearance. A separate
shadow problem may remain even when the tape is excluded.

For each existing terminal frame, use the same reference and pose in both
variants. Both variants run GrabCut with identical random seeds and search
settings. The second variant additionally excludes small, flat blue regions
near the original floor line. Vertically extended body tape is retained, and
the projected box interior is protected from this exclusion. Save the actual
exclusion mask, contour change and clearance shift. Do not promote the variant
without checking that it does not erase real corners or body tape.

Generate blind contact-review sheets for all 238 native GT episodes and the
120 original GT/Stage1/Stage2 query videos. Each case shows the left and right
reference, first terminal and last terminal appearance. Automatic outcomes and
angle estimates are deliberately absent from these sheets to avoid anchoring
the contact review on the detector's current answer. Ambiguous tiny gaps and
cropped corners require the full native-resolution case review.

The review labels concern actual visible contact, not center displacement or
a mandatory ratio of endpoint rises. Contact labels alone do not establish
the full action criterion: inclination and the complete final 0.3-second
interval must still be checked independently. Debug output stays in outputs.

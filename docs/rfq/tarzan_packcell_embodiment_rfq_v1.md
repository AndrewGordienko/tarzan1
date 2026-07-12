# Tarzan PackCell Embodiment RFQ - Application Engineering Packet v1

Date: 12 July 2026

Tarzan is selecting a single-arm embodiment for a teach-once small-sortable ecommerce packing cell. This packet requests written application-engineering confirmation; it is not a purchase order.

## Frozen task envelope

- Fifty deterministic development layouts; confirmation layouts remain sealed.
- Maximum conservative TCP/flange reach proxy: 1.148 m.
- Carton interior: 400 x 300 x 250 mm.
- Objects: 50 x 50 x 50 mm, 120 x 80 x 60 mm, and 250 x 150 x 100 mm representative shapes.
- Maximum workpiece mass: 8.0 kg.
- Minimum usable external jaw opening: 106 mm. Preferred procurement margin: 110-120 mm.
- Tool plus workpiece payload is evaluated with a 20% margin.
- Workpiece center-of-mass offset: up to 200 mm from the tool flange/TCP model.
- Current static finger-platform moment proxy: 23.1 Nm before custom-finger mass and dynamic acceleration.
- Minimum non-contact clearance: 3 mm; insertion must clear carton walls and rim.
- Object force limit: 80 N unless item metadata permits more. Fragile items require lower force, compliant tooling, suction, or escalation.
- Provisional motion target for quotation: 1.5 m/s2 nominal acceleration, 2.0 m/s2 peak, and 12 seconds per item. Please return a manufacturer-supported cycle estimate instead of treating these targets as validated.

## Arm application questions

Please evaluate UR15 and, if necessary, UR20 for all pick, pregrasp, lift, box-corner, deepest-placement, and staging poses. Confirm in writing:

1. Valid payload and center-of-gravity envelope for tool plus 8 kg workpiece with 20% margin.
2. Wrist force/torque validity at maximum extension and the declared CoG.
3. Acceleration, speed, and orientation derating.
4. Joint, singularity, collision, and stopping-distance constraints relevant to the cell.
5. Expected cycle time and whether UR20 provides a material benefit over UR15.
6. Supported force/torque interfaces, safety integration, field service, and lead time.
7. CAD, URDF, inertial, collision, payload-curve, and control-interface data available for simulation.

## End-effector application questions

### Custom-finger OnRobot 2FG14

The default 105 mm external opening does not pass. Please propose a manufacturer-supported finger configuration providing at least 106 mm usable opening, preferably 110-120 mm, while confirming:

- Minimum closure for 30-50 mm objects.
- Custom-finger dimensions, mass, center of gravity, inertia, and bending moment.
- Force transmission and allowable acceleration at an 8 kg workpiece.
- Compliance with the 25 Nm Y and 30 Nm X platform limits.
- Controllable force range for fragile packaged goods.
- Carton-entry and box-rim collision envelope.
- CAD, inertial, collision, and control-interface data with licensing terms.

### Hybrid jaws and suction

Please identify a supported vacuum tool or combined architecture for flat sealed cardboard/plastic parcels up to 8 kg. Provide cup geometry, seal-area requirements, porosity limits, vacuum reserve, acceleration derating, tool mass/CoG, seal-loss detection, and safe recovery behavior. Unknown or porous surfaces must fall back to jaws or human handling.

## Simulation and data requirements

- Official CAD plus version/revision identifiers.
- URDF or equivalent kinematic description where available.
- Link/tool masses, centers of gravity, and inertias.
- Collision geometry distinct from visual geometry.
- Joint, actuator, force, speed, acceleration, and payload limits.
- TCP/contact-frame definitions and finger/suction contact geometry.
- Control protocol documentation and sample integration code.
- Explicit license terms allowing internal simulation, derived MJCF generation, and benchmark publication.

## Requested written outcome

Please return one of: UR15 valid; UR20 required; custom 2FG14 valid; hybrid tool required; or no supported configuration. Include all boundary conditions and assumptions. Tarzan will not relax the 106 mm or 8 kg gates to fit a preferred component.

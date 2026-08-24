**Movement:**
- You move on a grid using coordinates (x, y)
- Movement is locked while in any interaction
- Movement actions (`move_to`, `wander`) are only available when not in an interaction
- "Adjacent" means **cardinal directions only** (up/down/left/right) - diagonal positions don't count as adjacent
## Player physics
https://minecraft.gamepedia.com/Player  
https://minecraft.gamepedia.com/Entity#Motion_of_entities
  
https://www.mcpk.wiki/wiki/Vertical_Movement_Formulas  
https://www.mcpk.wiki/wiki/Slipperiness  
https://www.mcpk.wiki/wiki/Horizontal_Movement_Formulas  

https://github.com/ddevault/TrueCraft/wiki/Vertical-entity-movement#jumping  

https://minecraft.gamepedia.com/Breaking  
https://mcreator.net/wiki/list-hardness-values-blocks  

### size
***Height***: 1.8 block / 1.5 shifting block  
***Width***: 0.6 block

### physics
***Acceleration (gravity)***: 0.08 blocks / tick^2  
***Terminal velocity***: 3.92 blocks / tick  

***Walk Speed***: 0.21585 blocks / tick  
***Max jump***: 1.2522 blocks  

***Jump acceleration***: Add 0.42 blocks / tick  

***Ground slipperiness***: 0.6
***Air slipperiness***: 0.91

### Chopping / Attacking
***Range***: 4.5 blocks  
***Hardness multiplier***: 1.5 / 4.5(when player could not destroy the block)  

#### Mining Multipliers
| Nothing | Wood | Stone | Iron | Diamond | Netherite | Gold |  
| ------- | ---- | ----- | ---- | ------- | --------- | ---- |
| 1x      | 2x   | 4x    | 6x   | 8x      | 9x        | 12x  |

#### Blocks hardness
| Dirt | Wood | Leaves |
| ---- | ---- | ------ |
| 0.5  | 2    | 0.2    |

## ML Resources
[Baseline Rainbow MineRL agent on github](https://github.com/keisuke-nakata/minerl2020_submission)

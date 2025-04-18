# Analysis - Category - Create calculate field

IF [Satisfaction Level]>=0.72 AND [Last Evaluation]>=0.80 THEN "Winners"

ELSEIF [Satisfaction Level]<=0.10 AND [Last Evaluation]>=0.75 THEN "Dis-Satisfied" 

ELSEIF [Satisfaction Level]>0.35 AND [Last Evaluation]<=0.57 THEN "Misfits"

ELSE "Others"

END


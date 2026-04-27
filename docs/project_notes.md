
'-' = Question
'+' = Anwser 
------------------------------------------------------------------------------
1-
df.drop(columns=['Machine_ID'],axis=1,inplace=True) 
- why dropped before correlation?

+ useless for correlation and expensive to encode the strings to floatpoints
------------------------------------------------------------------------------
 2-
 df = pd.get_dummies(df, columns=['Machine_Type'], dtype=int)
 df.fillna(0, inplace=True)
- why this approach?

+ because there are multiple machines and 0 that the machine doesnt use this feature and  
+ 0 is the standard and the model learn to ignore it,
+ random forest handle binary features very well 

+ otherwise i would have to use a model (knn) to fill nulls 
+ which is resource-expensive and time-consuming
------------------------------------------------------------------------------
3-
df.drop(columns=['Remaining_Useful_Life_days',
                 'Operational_Hours'],axis=1,inplace=True)
- why both and not one?

+ kept 'Operational_Hours' and confirmed that model cheated by looking at it
------------------------------------------------------------------------------
4-
smote = SMOTE(random_state=42)

- why applied Oversample technique?

+ 94% of data is biased to 0 (dont need go maintainance),
+ while only 6% need
------------------------------------------------------------------------------
5-
top_corr = target_corr[1:11]

- why 1 not 0?

+ target is 0
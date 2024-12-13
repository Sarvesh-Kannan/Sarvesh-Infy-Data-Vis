import pandas as pd
import matplotlib.pyplot as plt

feature_importances = pd.DataFrame({
    'Feature': [
        'ManagerID', 'DeptID', 'SpecialProjectsCount', 'PositionID', 
        'Position', 'DateofHire', 'DOB', 'EmpID', 'Zip', 'ManagerName', 
        'LastPerformanceReview_Date', 'EngagementSurvey', 'HispanicLatino', 
        'State', 'Department', 'Absences', 'Employee_Name', 'RaceDesc', 
        'RecruitmentSource', 'PerformanceScore', 'GenderID', 'EmpSatisfaction',
        'PerfScoreID', 'CitizenDesc', 'Sex', 'EmpStatusID', 'TermReason', 
        'DateofTermination', 'MaritalDesc', 'DaysLateLast30', 'MarriedID', 
        'MaritalStatusID', 'EmploymentStatus', 'FromDiversityJobFairID', 
        'Termd'
    ],
    'Importance': [
        0.255670, 0.203228, 0.057832, 0.053016, 0.046381, 0.044923, 0.039783, 
        0.038584, 0.034019, 0.031534, 0.023366, 0.022640, 0.022618, 0.016472, 
        0.015357, 0.014598, 0.014407, 0.011104, 0.007197, 0.006416, 0.005877, 
        0.005796, 0.004915, 0.003252, 0.002911, 0.002806, 0.002704, 0.002528, 
        0.002109, 0.002063, 0.001886, 0.001817, 0.000985, 0.000652, 0.000552
    ]
})

feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.pie(
    feature_importances['Importance'], 
    labels=feature_importances['Feature'], 
    autopct='%1.1f%%', 
    startangle=140, 
    textprops={'fontsize': 8},
    colors=plt.cm.tab20.colors
)

plt.title("Feature Contributions to Salary Prediction")
plt.axis('equal')  # Ensures the pie chart is a circle
plt.tight_layout()
plt.show()

from summarizers import Summarizers

summ = Summarizers(device="cuda")

content = "E-Time Tool - How to Get Managed by Employees. How to Apply for Leave. You Can Share Your Screen Now. Manager Delegations - What You Should Know. Second Option Timecard"

print(summ(content))

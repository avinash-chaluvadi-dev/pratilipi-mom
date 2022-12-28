**authorize**

This app specifies the generalized environments underneath user transection mechanism and login, logout, refresh access token, etc.

---

## logout mechanism:

whil refer to the Authorizeuser_sessionprohibition - domainid - is_blocked : true - email id - session token
Note: if all the above records are available on top of this table then will not allow the user to login with same sestoken.
will request user to refresh a new token

---

## refresh token mechanism:

- Authorizeuser_sessiontransaction update transectionlock : 0
- Generate new token
- block the previous token
- Update the existing Authorizeuser_customuser table sestoken = new token
- Create a new record with fresh token Authorizeuser_sessiontransaction table

Note: if all the above records are available on top of this tables then will allow the user to login with newly generated sestoken.

---

## New user create mechanism:

- Authorizeuser_customuser will update user provided data and create a new user record on top of database.

---

## login mechanism:

- Authorizeuser_customuser will update the newly generated token on top of sestoken column
- Authorizeuser_sessiontransaction will create a new transection record this table with transectionlock : 1

## model information:

- UserManager
- CustomUser
- SessionTransaction
- SessionProhibition

## serializer information:

- AuthorizeNewUserRegisterSerializer
- AuthorizeSessionTransactionSerializer
- AuthorizeSessionProhibitionSerializer

## related generalized functions : boiler_plage: utility

- get_sestokens
- set_session_prohibition_dtls
- set_sessiontransection_dtls
-

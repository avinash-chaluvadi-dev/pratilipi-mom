# -*- coding: utf-8 -*-
"""
    @Author         : WorkOS
    @Resource       : @AH40222 : Pratilipi Applicaiton : STARS Team
    @Purpose        : Generalized package which will help us to create the Jira issues and other stuff based on the MLOPs finalized votc reviews
    @Description    : At the end the created Jira will have the additional and complete and additional metadata, like app versions, OS and device data, for each review
                      Get the complete review history, etc.
    @Date           : 11-02-2022
    @Last Modified  : 15-02-2022

"""
import datetime
import json
import logging
import time
import traceback

import requests
import urllib3
import yaml
from requests.auth import HTTPBasicAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import logging as lg

logger = lg.getLogger("file")


class Jiraconfigurationcoreutils:
    # System configuration values:
    _dataSet = {}

    def __init__(
        self,
        base_config_path,
        log_format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        log_level="INFO",
        log_interval=5,
        **args,
    ):
        logging.basicConfig(format=log_format, level=log_level.upper())

        self._config_path = base_config_path
        self._user_agents = [
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322)",
            "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
        ]
        self._appName = ""
        self._appID = ""

        print("=================args:project:dtls:", args)

        self._jiraPojectKeyID = args["project_key"]
        self._jiraPojectID = args["project_id"]
        self._jiraParentKeyID = args["parent_key"]
        self._jiraParentID = args["parent_id"]
        self._jiraIssueTypeID = args["issue_type_id"]

        self._response = {}
        self._envType = ""
        self._envUnitType = ""
        self.base_userPassword = ""
        self.base_certificate_path = ""
        self.base_certificate_status = False
        self._base_request_url = ""
        self._url = ""
        self._session_request_type = "GET"
        self._auth = HTTPBasicAuth("", "")
        self._request_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._custom_metadtls = ""
        self._request_params = {}
        self._request_payload = {}
        self._jira_summary_type = "issue"
        self._jira_issue_type = {
            "Capability",
            "Defect",
            "Epic",
            "Initiative",
            "Risk",
            "Roll-up",
            "Sprint Goal",
            "Story",
            "Sub-task SUB-TASK",
            "Task",
            "Technical task SUB-TASK",
            "Test Case",
        }
        self._jira_Priorities_type = {
            "On-Cycle",
            "Off-Cycle",
            "Emergency",
            "Unassigned",
            "3-Normal",
            "4-Low",
            "2-High",
            "Medium",
            "high",
            "very high",
        }
        self._jira_label_types = "test-jira-votc-lable,"
        self._jira_component_types = "test-jira-votc-component,"

        logger.info(f"Initialised: {self.__class__.__name__}" f"('{self._url}')")

    # Request url creation
    def request_jira_url(self):
        try:
            # JIRA
            if (self._envType in "uat" and self._envUnitType in "JIRA") or (
                self._envType in "prod" and self._envUnitType in "JIRA"
            ):
                self._url = f"https://{self._base_request_url}/rest/api/2/{str(self._custom_metadtls)}"

                print(
                    "=======> Jira-Env:{0}:URL:preparing: ".format(self._envType),
                    self._url,
                )

            logger.info(f"Jira url : {self._url}")
        except Exception as err:
            print(traceback.format_exc())
            logger.warning(
                f"Initialised: {self.__class__.__name__}"
                f"('{self._url}')"
                f"Exception {0}".format(err)
            )

    # Request-POST responce
    def jira_session_request(self):
        try:

            self._auth = HTTPBasicAuth(self._appID, self.base_userPassword)

            print("PAYLOAD:::", self._request_payload)
            # exit(0)

            if self.base_certificate_status:
                self._response = requests.request(
                    self._session_request_type,
                    self._url,
                    data=self._request_payload,
                    headers=self._request_headers,
                    auth=self._auth,
                    verify=self.base_certificate_path,
                )
            else:
                self._response = requests.request(
                    self._session_request_type,
                    self._url,
                    data=self._request_payload,
                    headers=self._request_headers,
                    auth=self._auth,
                    verify=self.base_certificate_status,
                )
            time.sleep(3)
            print("=====================================:done")
            print("=========================data:::", self._response)
            print("=========================TEXT::", self._response.text)
            print("DATA:", json.loads(self._response.text))
            print("jira_data_status_code:", self._response.status_code)

            if (
                int(self._response.status_code) == 400
                or int(self._response.status_code) >= 400
            ):  # error to connect jira server.
                self._response = "error"  # smtp- error found drift :: -- Email* - admin | Ethan and Platform
                print(
                    "=====================================:ERROR - FOUND:",
                    self._envType,
                    self._envUnitType,
                    self._appName,
                )
            else:  # Data Available
                print(
                    "=====================================:FOUND:",
                    self._envType,
                    self._envUnitType,
                    self._appName,
                )
                self._response = json.loads(
                    self._response.text
                )  # "session connected successfully"  # smtp- no-data found drift:: -- Email* - admin | Ethan and Platform

        except Exception as err:
            print(traceback.format_exc())
            logger.warning(
                f"Initialised: {self.__class__.__name__}"
                f"('{self._url}')"
                f"Exception {0}".format(err)
            )

    # GET REUSING CONFIGURATIONS
    def get_jira_configuration(self, *args, **kwargs):
        try:
            import json

            with open(r"{0}".format(self._config_path)) as conf_management:
                jiraAppconfig_list = yaml.load(conf_management, Loader=yaml.FullLoader)
                datalen = len(
                    jiraAppconfig_list.get("jiraconfigurationmngservicedtls")
                    .get("execution")
                    .get("service_request")
                )
                print("LN", datalen)
                # print("YAML:CONFIG:", str(jiraAppconfig_list.get('jiraconfigurationmngservicedtls').get('execution').get('service_request')))

                # Calling:data:appbot:
                dataJiraObj = {}

                # loop over data sources available:
                for item in range(0, datalen):
                    jiraConfigDetails = (
                        jiraAppconfig_list.get("jiraconfigurationmngservicedtls")
                        .get("execution")
                        .get("service_request")[item]
                    )
                    if jiraConfigDetails.get("applicable"):
                        # Jira service configurations
                        print("Core Jira details config ...\n")

                        if (
                            str("-uat")
                            in str(jiraConfigDetails.get("service_type")).lower()
                        ) or (
                            str("-prod")
                            in str(jiraConfigDetails.get("service_type")).lower()
                        ):
                            self._envUnitType = "JIRA"
                            self._envType = (
                                str(jiraConfigDetails.get("service_type"))
                                .lower()
                                .split("-")[1]
                            )
                            self._appID = jiraConfigDetails.get("parameters").get(
                                "sericeId"
                            )
                            self._appName = jiraConfigDetails.get("parameters").get(
                                "host_name"
                            )
                            self._jiraPojectKeyID = jiraConfigDetails.get(
                                "parameters"
                            ).get("key")
                            self._base_request_url = jiraConfigDetails.get(
                                "parameters"
                            ).get("host_name")
                            self.base_certificate_path = jiraConfigDetails.get(
                                "parameters"
                            ).get("certificate")
                            self.base_certificate_status = jiraConfigDetails.get(
                                "parameters"
                            ).get("chk_certificate")
                            self.base_userPassword = jiraConfigDetails.get(
                                "parameters"
                            ).get("password")

                            dataJiraObj[self._envType] = dict(self.__dict__)
                return dataJiraObj
        except KeyboardInterrupt:
            logger.error("Keyboard interrupted")
            print(traceback.format_exc())
        except Exception as e:
            logger.error(f"Something went wrong: {e}")
            print(traceback.format_exc())

    # Issue:title:summary
    def _generate_summary(self):
        try:

            if self._jira_summary_type in "issue":
                return "Summary - " + "{date:%Y-%m-%d %H:%M}".format(
                    date=datetime.datetime.now()
                )

        except Exception as err:
            print(traceback.format_exc())
            print("Exception {0}".format(err))

    # Issue:description
    def _generate_description(self, **args):
        try:

            # issue-jira
            if self._jira_summary_type in "issue":
                runtimedesc = (
                    # "Creating of an issue//nusing project keys and issue type names using the REST API"
                    args["data"]
                    + " "
                    + "{date:%Y-%m-%d %H:%M}".format(date=datetime.datetime.now())
                )
                return runtimedesc

        except Exception as err:
            print(traceback.format_exc())
            print("Exception {0}".format(err))

    # jira-labels
    def _set_jira_lables(self):
        try:

            # issue-jira
            if self._jira_label_types != None and len(self._jira_label_types) > 0:

                return self._jira_label_types

        except Exception as err:
            print(traceback.format_exc())
            print("Exception {0}".format(err))

    # components-jira
    def _set_jira_components(self):
        try:

            # issue-jira
            if (
                self._jira_component_types != None
                and len(self._jira_component_types) > 0
            ):

                return self._jira_component_types

        except Exception as err:
            print(traceback.format_exc())
            print("Exception {0}".format(err))

    # Issue:other:core-data
    def _generate_issue_data(self, **args):
        # Build the JSON to post to JIRA
        # Issue:payload:data
        try:
            # return True
            if "sub-task" not in args["key"]:
                # print("=========If:>>", args["key"])
                votc_jira_issue_json_data = """
                {
                    "fields":{
                        "project":{
                            "key":"%s"
                        },
                        "summary": "%s",
                        "issuetype":{
                            "name":"%s"
                        },
                        "description": "%s",
                        "labels": ["%s"],
                        "components": [{ "name": "%s"}]
                    }

                } """ % (
                    self._jiraPojectKeyID,
                    self._generate_summary(),
                    self._jira_issue_type,
                    self._generate_description(data=args["data"]),
                    self._jira_label_types[:-1],
                    self._jira_component_types[:-1],
                )
            else:
                # print("=========Else:>>", args["key"])
                votc_jira_issue_json_data = """ {
                    "fields":{
                        "project":{ "key":"%s"},
                        "parent":{ "key": "%s"},
                        "summary": "%s",
                        "description": "%s",
                        "issuetype": { "id": "%s" }
                        }} """ % (
                    self._jiraPojectKeyID,
                    self._jiraParentKeyID,
                    self._generate_summary(),
                    self._generate_description(data=args["data"]),
                    self._jiraIssueTypeID,
                )

            return votc_jira_issue_json_data

        except Exception as err:
            print(traceback.format_exc())
            print("Exception {0}".format(err))

    # Issue:description
    def _create_jira_issue(self, **args):
        try:
            # print("========inside.", args)
            self._request_payload = self._generate_issue_data(
                key=args["type"], data=args["data"]
            )
            # print(self._request_payload)
            # exit(0)
            self._custom_metadtls = "issue"
            self._session_request_type = "POST"
            self.request_jira_url()
            self.jira_session_request()

        except Exception as err:
            print(traceback.format_exc())
            print("Exception {0}".format(err))  # create issue:

    # Issue:description
    def _create_jira_bulk_issue(self, **args):
        try:
            # print("========inside.", args)
            bulk_issue = {
                "issueUpdates": [],
            }
            bulk_action_issues = list()
            if args["items"] != None:
                for dtls in args["items"]:
                    _request_payload = self._generate_issue_data(
                        key=args["type"], data=str(dtls)
                    )
                    bulk_action_issues.append(json.loads(_request_payload))
                    _request_payload = ""

            # print("BULCK-ISSUES", bulk_action_issues)
            bulk_issue["issueUpdates"] = bulk_action_issues
            time.sleep(2)

            print("=============[query_set]=========", json.dumps(bulk_issue))

            self._request_payload = json.dumps(bulk_issue)
            self._custom_metadtls = "issue/bulk"
            self._session_request_type = "POST"
            self.request_jira_url()
            time.sleep(2)
            self.jira_session_request()

            print("FINAL::RESPONSE::", self._response)

        except Exception as err:
            print(traceback.format_exc())
            print("Exception {0}".format(err))  # create issue:

        return self._response

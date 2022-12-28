# -*- coding: utf-8 -*-
"""
    @Author         : HIVE TEAM
    @Purpose        : REST_API Application views***
    @Description    : 
    @Date           : 05-08-2021
    @Last Modified  : 05-08-2021
"""

import json
import logging as lg
from collections import OrderedDict
from typing import Union

import magic
from apscheduler.schedulers.background import BackgroundScheduler
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import generics, status
from rest_framework.decorators import action, api_view
from rest_framework.generics import RetrieveUpdateAPIView
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from authorize.utils.api_permissions import (ConfigScreenAPIPermission,
                                             FeedbackLoopAPIPermission,
                                             FileUploadAPIPermission,
                                             InsightDashboardAPIPermission,
                                             MoMAPIPermission)
from boiler_plate.corejira.core_jira_utility import Jiraconfigurationcoreutils
from boiler_plate.utility import constants, utils
from boiler_plate.utility.html_to_pdf_converter import render_to_pdf
from boiler_plate.utility.ml_serve import serve_ml_models
from module import specs
from module.utils import generate_mom

from .models import (ConsolidateModelsData, FeedBackLoop, File, JIRADetails,
                     JiraTransaction, MeetingMetadata, Team)
from .serializers import (ConsolidateModelsDataSerializer,
                          FileUploadSerializer, JiraTransactionSerializer,
                          MeetingMetadataSerializer, TeamSerializer)
from .utils import (dashboard_file_status_counts, dashboard_label_counts,
                    dashboard_label_details, detailed_view_filter,
                    parse_string_datetime)

logger = lg.getLogger("file")

from rest_framework.permissions import IsAuthenticated


@api_view(["GET"])
def status_check(request: object, *args: list, **kwargs: dict) -> Response:
    """
    API wrapper for checking the status of ML models
    """

    # Fetching model_name from kwargs
    model_name = kwargs.get("model")
    logger.info(f"Checking status of {model_name.capitalize()} job")

    response_dict = json.loads(request.body)
    response_path = response_dict[list(response_dict.keys())[0]]
    logger.info(f"Loading {model_name.capitalize()} output json from {response_path}")
    logger.info(f"{model_name.capitalize()} output json loaded successfully")

    status_response = utils.model_status(file_path=response_path, model_name=model_name)
    logger.info(f"{model_name.capitalize()} status response --> {status_response}")

    return Response(status_response)


@api_view(["POST"])
def file_status_update(request):
    status = request.POST.get("status")
    file_name = request.POST.get("file")
    masked_request_id = request.POST.get("masked_request_id")
    backend_start_time = request.POST.get("backend_start_time")
    response = "".join(filter(None, [status, backend_start_time]))

    if masked_request_id:
        file_obj = get_object_or_404(File, masked_request_id=masked_request_id)
        # Updates either backend_start_time or status in api_file table
        if backend_start_time:
            file_obj.backend_start_time = backend_start_time
        elif status:
            file_obj.status = status
        file_obj.save()
        return Response(
            {
                "message": f"File status updated for {masked_request_id} with {response.strip()}"
            }
        )

    elif file_name:
        file_obj = get_object_or_404(File, file=file_name)
        # Updates either backend_start_time or status in api_file table
        if backend_start_time:
            file_obj.backend_start_time = backend_start_time
        elif status:
            file_obj.status = status
        file_obj.save()
        return Response(
            {"message": f"File status updated for {file_name} with {response.strip()}"}
        )


@api_view(["GET"])
def data_filter(request: object, *args: list, **kwargs: dict) -> Response:

    # Creates s3_utils object to interact with S3 using boto3
    s3_utils = utils.S3Utils()
    # Fetches request_id and corresponding file object from api_file table
    request_id = kwargs.get("request_id")
    file_obj = get_object_or_404(File, masked_request_id=request_id)

    # Extrcating mom specs from specs.yaml file
    mom_specs = specs.get_mom_specs()
    mom_input = mom_specs.get("input")
    mom_output = mom_specs.get("output")
    output_file_name = mom_specs.get("output_file_name")

    # Get's S3 base path corresponding to request_id
    base_path = utils.get_file_base_path(
        file_obj.get_file(),
        mom_input,
        file_obj.get_file_name(),
    )
    mom_output_prefix = f"{base_path}/{mom_output}/{output_file_name}"
    # Checks whether mom.json file exist or not using api_mom table
    output_exist = s3_utils.prefix_exist(file_prefix=mom_output_prefix)
    # Loads mom response from S3 into mom_response variable
    mom_response = utils.get_input_file_as_dict(
        file_path=mom_output_prefix, output_exist=output_exist
    )

    # Get's data_filter payload from request object
    data_filter_payload = json.loads(request.body)
    if not bool(data_filter_payload):
        response = {
            constants.STATUS_KEY: status.HTTP_200_OK,
            constants.DATA_KEY: mom_response,
        }
        return Response(**response)
    else:
        filtered_response = detailed_view_filter(
            mom_response=mom_response, data_filter_payload=data_filter_payload
        )
        response = {
            constants.STATUS_KEY: status.HTTP_200_OK,
            constants.DATA_KEY: filtered_response,
        }
        return Response(**response)


class FileViewSet(ModelViewSet):
    permission_classes = [FileUploadAPIPermission]
    http_method_names = [
        "get",
        "post",
        "put",
        "patch",
        "head",
        "options",
        "trace",
    ]
    queryset = File.objects.exclude(status=constants.CANCELLED_DB).select_related(
        "metadata"
    )
    serializer_class = FileUploadSerializer
    lookup_field = "masked_request_id"

    def get_serializer_context(self):
        """
        Extra context provided to the serializer class.
        """
        context = super(FileViewSet, self).get_serializer_context()
        context.update({"request": self.request})
        return context

    def get_queryset(self):
        file_name = self.request.query_params.get("file")
        if file_name:
            self.queryset = self.queryset.filter(file=file_name)
        else:
            self.queryset = self.queryset.filter(user=self.request.user)
        return self.queryset

    def create(self, request: HttpRequest, *args: list, **kwargs: dict) -> Response:
        """
        View used for uploading a new file
        """
        file_object = request.data.get("file")
        extension_type = magic.from_buffer(file_object.read(2048), mime=True)
        if extension_type not in constants.SUPPORTED_MIME_TYPES:
            logger.info(
                f"Currently Pratilipi tool doesn't support {extension_type} format of files"
            )
            supported_mime_type_string = ", ".join(constants.SUPPORTED_FILE_TYPES)
            response = {
                constants.STATUS_KEY: status.HTTP_400_BAD_REQUEST,
                constants.DATA_KEY: f"Unsupported file format '{extension_type}', Please try to upload one of the file types from {supported_mime_type_string}",
            }
            return Response(**response)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(
            serializer.data, status=status.HTTP_201_CREATED, headers=headers
        )

    @action(detail=True, methods=["PATCH"])
    def cancel_extraction(
        self, request: HttpRequest, masked_request_id: str = None
    ) -> Response:
        """
        This view will update the file status to cancelled
        """

        # Creates s3_utils object to interact with S3 using boto3
        s3_utils = utils.S3Utils()

        file_obj = self.get_object()
        file_obj.status = constants.CANCELLED_DB
        file_obj.save()

        # Deletes the file from S3 bucket
        s3_utils.delete_object(file_prefix=str(file_obj.file))

        # Creates cancel response string using constants module
        response_string = f"File extraction {constants.CANCELLED_DB}, successfully removed {file_obj.file} from {settings.AWS_STORAGE_BUCKET_NAME}"
        return Response({"message": response_string}, status=status.HTTP_200_OK)


@api_view(["GET"])
def process_meeting(request: HttpRequest, *args: list, **kwargs: dict) -> Response:
    """Process the meeting by calling all the ML models"""
    SCHEDULER_CONFIG = {
        "apscheduler.jobstores.default": {
            "class": "django_apscheduler.jobstores:DjangoJobStore"
        },
        "apscheduler.executors.processpool": {"type": "threadpool"},
        "apscheduler.job_defaults.coalesce": "false",
    }
    request_id = kwargs.get("request_id")
    scheduler = BackgroundScheduler(SCHEDULER_CONFIG)
    file_obj = get_object_or_404(File, masked_request_id=request_id)
    file_obj.status = "Processing"
    file_obj.save()
    scheduler.add_job(serve_ml_models, kwargs=kwargs)
    scheduler.start()
    return Response({"message": "process started"})


class MeetingMetadataRetrieveUpdateAPIView(RetrieveUpdateAPIView):
    queryset = MeetingMetadata.objects.all().select_related("meeting")
    serializer_class = MeetingMetadataSerializer
    permission_classes = [FeedbackLoopAPIPermission]

    def get_object(self):
        """
        Returns the object the view is displaying.

        You may want to override this if you need to provide non-standard
        queryset lookups.  Eg if objects are referenced using multiple
        keyword arguments in the url conf.
        """
        queryset = self.filter_queryset(self.get_queryset())

        # Perform the lookup filtering.
        self.lookup_field = "request_id"
        lookup_url_kwarg = self.lookup_url_kwarg or self.lookup_field

        assert lookup_url_kwarg in self.kwargs, (
            "Expected view %s to be called with a URL keyword argument "
            'named "%s". Fix your URL conf, or set the `.lookup_field` '
            "attribute on the view correctly."
            % (self.__class__.__name__, lookup_url_kwarg)
        )

        filter_kwargs = {self.lookup_field: self.kwargs[lookup_url_kwarg]}
        obj = get_object_or_404(
            queryset, meeting__masked_request_id=filter_kwargs["request_id"]
        )

        # May raise a permission denied
        self.check_object_permissions(self.request, obj)

        return obj


class GenerateMoMPDFAPIView(APIView):
    permission_classes = [MoMAPIPermission]

    def get(
        self, request: HttpRequest, *args: list, **kwargs: dict
    ) -> Union[HttpResponse, Response]:
        """GET API which generate PDF from the latest mom.json and send it as an attachment in the response"""

        request_id = kwargs.get("request_id")
        file_obj = get_object_or_404(File, masked_request_id=request_id)
        try:
            mom = generate_mom(request_id, file_obj)
            context = {
                "mom": mom["mom_entries"],
                "overview": mom["overview"],
                "metadata": MeetingMetadata.objects.get(
                    meeting__masked_request_id=request_id
                ),
                "date": file_obj.date,
            }
            pdf = render_to_pdf("mom.html", context)
            if pdf:
                response = HttpResponse(pdf, content_type="application/pdf")
                response["Content-Disposition"] = "attachment; filename=mom.pdf"
                return response
        except Exception as e:
            logger.error(
                "Execption {} occured while generating PDF for request ID {}".format(
                    e.__class__, request_id
                )
            )
            return Response(
                {"error": "something went wrong while generating PDF"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        logger.error("Could not generated PDF for request id {} ".format(request_id))
        return Response(
            {"error": "something went wrong while generating PDF"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class ConsolidateParticipantDashBoardBlenddatasets(generics.GenericAPIView):
    serializer_class = ConsolidateModelsDataSerializer
    permission_classes = [InsightDashboardAPIPermission]

    def get(self, request, *args, **kwargs) -> Response:
        # Get's start and end date from request params
        start_date = request.GET.get("start_date")
        end_date = request.GET.get("end_date")

        # Let's add log statements for further monitoring
        logger.info(
            f"Dashboard start date filter in format of {constants.DASHBOARD_DATE_FORMAT} is {start_date}"
        )
        logger.info(
            f"Dashboard end date filter in format of {constants.DASHBOARD_DATE_FORMAT} is {end_date}"
        )

        # Get's Dashboard details of a particular user
        try:
            # Initializes dictionary objects for storing file/label counts/details
            dashboard_details = OrderedDict()
            dashboard_details["dashboard_info"] = {}
            dashboard_details["team_info"] = OrderedDict()

            # Parses start_date and end_date to a django supported format
            parsed_start_date, parsed_end_date = parse_string_datetime(
                start_date, end_date
            )

            if request.user.is_admin:
                # Creates database tables objects by applying necessary filters
                file_object = (
                    File.objects.exclude(masked_request_id__exact="")
                    .exclude(status__exact=constants.CANCELLED_DB)
                    .filter(date__gte=parsed_start_date, date__lt=parsed_end_date)
                    .select_related()
                )
            elif request.user.is_sme:
                file_object = (
                    File.objects.exclude(masked_request_id__exact="")
                    .exclude(status__exact=constants.CANCELLED_DB)
                    .filter(user_id=request.user)
                    .filter(date__gte=parsed_start_date, date__lt=parsed_end_date)
                    .select_related()
                )

            # Let's add log statements for further monitoring
            user_role = request.user._role
            logger.info(f"Role of {request.user} is {user_role.capitalize()}")
            if start_date == end_date:
                logger.info(
                    f"Total number of files for {user_role} on {start_date} --> {len(file_object)}"
                )

            else:
                logger.info(
                    f"Total number of files for {user_role} from {start_date} to {end_date} --> {len(file_object)}"
                )

            # Creates database tables objects by applying necessary filters
            mom_object = FeedBackLoop.objects.filter(is_mom_exist=1).filter(
                request_id__in=file_object.values("masked_request_id")
            )
            consolidated_data = ConsolidateModelsData.objects.filter(
                file_id__in=mom_object.values("request_id")
            ).values()

            # Calls dashboard utility functions to get respective counts
            dashboard_file_status_counts(
                file_object=file_object, data_dict=dashboard_details
            )
            dashboard_label_counts(
                consolidated_data=consolidated_data, data_dict=dashboard_details
            )
            dashboard_label_details(
                consolidated_data=consolidated_data,
                data_dict=dashboard_details,
                mom_object=mom_object,
            )

            # Creates success dashboard result dictionary with status code as 201
            result = {"status": status.HTTP_201_CREATED}
            result["data"] = dashboard_details
            result["status"] = status.HTTP_200_OK
            return Response(status=result["status"], data=result["data"])

        except Exception as e:
            # Creates dashboard error result dictionary with status code as 500
            result = {"status": status.HTTP_500_INTERNAL_SERVER_ERROR}
            result["data"] = {}
            logger.error(
                f"Something went wrong while processing dash board api request...!\nstr{e}"
            )
            return Response(status=result["status"], data=result["data"])


class ApiJiraTransactionView(generics.GenericAPIView):
    serializer_class = JiraTransactionSerializer

    def post(self, request, *args, **kwargs):
        try:
            import re
            import time
            import traceback
            from pathlib import Path

            from boiler_plate.utility.orm_bulkcnfg import SensemakerbulkDBMnger

            """
            :Here will call the utility Pratilipi JIRA component
            :The defined jira custom property try to build and create the jira sub-task on top of defined user stories** project wise.
            :Add the end will save the Jira transection details in JIRA- Transection table.
            """

            result = {"status": status.HTTP_201_CREATED}
            base_config_path = f"{Path(__file__).resolve().parent.parent}/{constants.APP_CONFIG_PATH['jira']}"
            consolidate_jira_trans_mgr = SensemakerbulkDBMnger(chunk_size=100)
            if request.data != None and len(request.data) > 0:
                for titem in request.data:
                    try:
                        stime = time.strftime("%H:%M:%S:%MS", time.localtime())
                        team_dlts = JIRADetails.objects.get(team__name__iexact=titem)
                        if "-" in team_dlts.ticket_no and team_dlts.ticket_no != None:
                            pattern = re.compile("^[A-Z|a-z]*")
                            project_key = pattern.findall(str(team_dlts.ticket_no))[0]
                            parent_key = team_dlts.ticket_no

                            objutil = Jiraconfigurationcoreutils(
                                base_config_path,
                                project_key=project_key,
                                project_id="",
                                parent_key=parent_key,
                                parent_id="",
                                issue_type_id="10109",  # sub-task
                                # issue_type_id="10001",  #story
                            )

                            objutil.get_jira_configuration()
                            time.sleep(0.5)

                            res_jira_dtl = objutil._create_jira_bulk_issue(
                                type="sub-task", items=request.data.get(titem)
                            )

                            jira_dtls = JIRADetails.objects.only("id").get(
                                id=team_dlts.id
                            )

                            if res_jira_dtl.get("issues") != None:
                                for res_data in res_jira_dtl.get("issues"):
                                    consolidate_jira_trans_mgr.add(
                                        JiraTransaction(
                                            jira_issue_id=res_data.get("id"),
                                            jira_issue_key=res_data.get("key"),
                                            jira_issue_url=res_data.get("self"),
                                            jira_detail_id=int(team_dlts.id),
                                        )
                                    )

                            result["data"] = {
                                "project_key": project_key,
                                "parent_key": parent_key,
                                "Issues": res_jira_dtl.get("issues"),
                                "message": "New JIRA sub-task issues has been created on top of the above parent JIRA issue successfully !",
                            }
                            result["status"] = status.HTTP_201_CREATED

                            # bulk-add
                            consolidate_jira_trans_mgr.done()

                            etime = time.strftime("%H:%M:%S:%MS", time.localtime())

                    except Exception as err:
                        result["data"] = {
                            "error": f"Something went wrong while processing jira issues @pratilipi tool ! Exception {err}"
                        }
                        result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                        logger.error(f"Exception :: ApiJiraTransactionView :: str{err}")
            else:
                result["data"] = {
                    "error": f"The bad request submitted please check your submit query once again! before processing jira issues @pratilipi tool !"
                }
                result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
                logger.error(f"Exception :: ApiJiraTransactionView ::")

        except Exception as err:
            result["data"] = {
                "error": f"Something went wrong while processing jira issues @pratilipi tool ! Exception {err}"
            }
            result["status"] = status.HTTP_500_INTERNAL_SERVER_ERROR
            logger.error(f"Exception :: ApiJiraTransactionView :: str{err}")

        return Response(status=result["status"], data=result["data"])


class TeamViewSet(ModelViewSet):
    queryset = Team.objects.all()
    serializer_class = TeamSerializer
    # permission_classes = [ConfigScreenAPIPermission]

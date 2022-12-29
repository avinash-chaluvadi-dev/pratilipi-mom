#stage 1
FROM quay.apps.lz-np2.ent-ocp4-useast1.aws.internal.das/openshift-base-images/ubi8-nodejs16-react17-image-with-certs AS react-build
USER root
COPY  / /codebase
WORKDIR /codebase
RUN npm install
RUN npm run build

#stage 2
FROM quay.apps.lz-np2.ent-ocp4-useast1.aws.internal.das/openshift-base-images/ngnix-image-with-certs:latest
USER root
RUN rm -rf /usr/share/nginx/html/*
COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=react-build /codebase/build /usr/share/nginx/html/mom-webapp
EXPOSE 80
USER 1001
CMD ["nginx", "-g", "daemon off;"]
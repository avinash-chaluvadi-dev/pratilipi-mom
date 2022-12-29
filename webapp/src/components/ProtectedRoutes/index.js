import React from "react";
import Layout from "components/Layout";

const ProtectedRoutes = ({ children }) => {
    return <Layout>{children}</Layout>;
};
export default ProtectedRoutes;

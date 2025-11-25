import { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";
import { ROUTES } from "./routes";
import { Loader } from "src/components";

const HomePage = lazy(() => import("../pages/HomePage"));
const AboutPage = lazy(() => import("../pages/AboutPage"));
const YachtDetailsPage = lazy(() => import("../pages/YachtDetailsPage"));
const NotFoundPage = lazy(() => import("../pages/NotFoundPage"));
const AddYachtPage = lazy(() => import("../pages/AddYachtPage")); 

const AppNavigator = () => {
  return (
    <Suspense fallback={<Loader />}>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path={ROUTES.ABOUT} element={<AboutPage />} />
        
        <Route path={ROUTES.ADD_YACHT} element={<AddYachtPage />} />
        
        <Route path={ROUTES.YACHT_DETAILS} element={<YachtDetailsPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </Suspense>
  );
};

export default AppNavigator;

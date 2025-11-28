import { Stack } from "@mui/material";
import Header from "../Header";
import { StyledWrapper } from "./styled";
import { useAuth } from "src/context/authContext";
import { useEffect } from "react";

const PageWrapper = ({ children }) => {

  return (
    <StyledWrapper>
      <Header />
      {children}
    </StyledWrapper>
  );
};

export default PageWrapper;

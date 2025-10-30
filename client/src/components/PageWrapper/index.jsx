import { Stack } from "@mui/material";
import Header from "../Header";
import { StyledWrapper } from "./styled";
import { useAuth } from "src/context/authContext";
import { useEffect } from "react";

const PageWrapper = ({ children }) => {
  const { setUser } = useAuth();

  useEffect(() => {
    const storedToken = localStorage.getItem("token");
    const storedUserId = localStorage.getItem("userId");
    if (storedToken && storedUserId) {
      setUser({ token: storedToken, id: storedUserId });
    }
  }, [setUser]);

  return (
    <Stack>
      <Header />
      <StyledWrapper>{children}</StyledWrapper>
    </Stack>
  );
};

export default PageWrapper;

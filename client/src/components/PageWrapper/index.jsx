import { Stack } from "@mui/material";
import Header from "../Header";
import { StyledWrapper } from "./styled";

const PageWrapper = ({ children }) => {
  return (
    <Stack>
      <Header />
      <StyledWrapper>{children}</StyledWrapper>
    </Stack>
  );
};

export default PageWrapper;

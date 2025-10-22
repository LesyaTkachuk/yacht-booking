import { FC } from "react";

import { CircularProgress, Stack } from "@mui/material";

const Loader = ({ text }) => {
  return (
    <Stack
      width={"100%"}
      justifyContent={"center"}
      alignItems={"center"}
      padding={4}
      minHeight={"200px"}
    >
      <CircularProgress />
    </Stack>
  );
};

export default Loader;

import { FC } from "react";

import { Typography } from "@mui/material";
import { BorderedContainer } from "./styled";

const SearchBar = () => {
  return (
    <BorderedContainer>
      <Typography>{text}</Typography>
    </BorderedContainer>
  );
};

export default SearchBar;

import { ChangeEvent, FC } from "react";

import CloseIcon from "@mui/icons-material/Close";
import SearchIcon from "@mui/icons-material/Search";
import {
  InputAdornment,
  ListSubheader,
  TextField as MuiTextField,
} from "@mui/material";

import { StyledIconButton } from "./styled";

const SearchField = ({
  searchValue = "",
  onSearchChange = () => {},
  clearSearch = () => {},
  className,
}) => {
  return (
    <ListSubheader sx={{ paddingTop: "20px" }} className={className}>
      <MuiTextField
        size="small"
        autoFocus
        placeholder="Type to search..."
        fullWidth
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
          endAdornment: (
            <InputAdornment position="end">
              <StyledIconButton onClick={clearSearch} disableRipple>
                <CloseIcon />
              </StyledIconButton>
            </InputAdornment>
          ),
          autoComplete: "off",
          autoCorrect: "off",
          autoCapitalize: "off",
          spellCheck: "false",
        }}
        value={searchValue}
        onChange={onSearchChange}
        onKeyDown={(e) => {
          if (e.key !== "Escape") {
            // Prevents auto-selecting item while typing on Search field (default Select behavior)
            e.stopPropagation();
          }
        }}
      />
    </ListSubheader>
  );
};

export default SearchField;

"use client";

import { useState } from "react";
import { Controller, useFormContext } from "react-hook-form";

import { CircularProgress, Stack } from "@mui/material";


import SearchField from "./SearchField";
import SelectItem from "./SelectItem";
import {
  HiddenMenuItem,
  StyledMenuItem as SelectMenuItem,
  StyledTypography,
  StyledTextField,
} from "./styled";

import useIntersectionObserverRef from "./useIntersectionObserverRef";
import useSearch from "./useSearch";

// to enable Search in default Select component just pass withSearch={true}
// to enable Search in custom Select component build SearchField directly, use useSearch hook and  pass withSearch={SearchField} (refer to example AssetSelect)
const Select = ({
  name,
  children,
  renderValue,
  items,
  label,
  placeholder = "Select an option",
  multiple = false,
  errorMessage,
  itemTooltipTitle,
  hasNextPage,
  isFetching,
  fetchNextPage,
  withSearch,
  width,
  isRequired = true,
  onChange: onSelectChange,
  ...props
}) => {
  const [menuWidth, setMenuWidth] = useState(400);
  const { control } = useFormContext();

  const fallbackRenderValue = (value) => {
    const item = items?.find((item) => item.value === value);
    return (
      <StyledTypography
        variant="body1"
        color={value ? "secondary.contrastText" : "secondary"}
      >
        {item?.title || value || placeholder}
      </StyledTypography>
    );
  };

  const { filteredItems, searchValue, onSearchChange, clearSearch } =
    useSearch(items);

  const lastEntryRef = useIntersectionObserverRef(
    isFetching,
    hasNextPage,
    fetchNextPage
  );

  return (
    <Controller
      control={control}
      name={name}
      rules={{ required: isRequired ? errorMessage ?? placeholder : false }}
      render={({ field: { onChange, onBlur, value, ref } }) => (
        // @ts-ignore
        <StyledTextField
          select
          name={name}
          inputRef={(r) => {
            ref(r);

            if (r?.node?.clientWidth) {
              setMenuWidth(r.node.clientWidth);
            }
          }}
          slotProps={{
            select: {
              displayEmpty: true,
              multiple: multiple,
              renderValue: (value) => {
                return renderValue
                  ? renderValue(value)
                  : fallbackRenderValue(value);
              },
              onClose: clearSearch,
              MenuProps: {
                anchorOrigin: { vertical: "bottom", horizontal: "left" },
                transformOrigin: { vertical: "top", horizontal: "left" },
                autoFocus: false,
                PaperProps: {
                  style: {
                    marginTop: 5,
                    borderRadius: 8,
                    maxHeight: 400,
                    padding: 0,
                    minWidth: menuWidth,
                  },
                },
                MenuListProps: {
                  style: {
                    padding: 0,
                  },
                },
              },
            },
          }}
          label={label}
          width={width}
          onChange={(e) => {
            if (onSelectChange) {
              onSelectChange(e);
            }

            onChange(e);
          }}
          onBlur={onBlur}
          value={value}
          {...props}
        >
          {withSearch && items && (
            <SearchField
              searchValue={searchValue}
              onSearchChange={onSearchChange}
              clearSearch={clearSearch}
            />
          )}
          <HiddenMenuItem value="" disabled />
          {children ||
            filteredItems?.map((option) => {
              const { value, icon, title, withTooltipTitle } = option;

              return (
                <SelectMenuItem key={value} value={value} disableRipple>
                  <SelectItem
                    icon={icon}
                    title={title || value}
                    withTooltipTitle={withTooltipTitle}
                    tooltipTitle={itemTooltipTitle}
                    my={1}
                  />
                </SelectMenuItem>
              );
            })}
          {isFetching && (
            <Stack direction="row" justifyContent="center" py={1}>
              <CircularProgress />
            </Stack>
          )}
          <div ref={lastEntryRef} />
        </StyledTextField>
      )}
    />
  );
};

export default Select;

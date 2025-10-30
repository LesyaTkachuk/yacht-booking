"use client";

import { useState } from "react";

const useSearch = (items) => {
  const [searchValue, setSearchValue] = useState("");
  if (!items) return {};

  const filteredItems = items?.filter(
    ({ value, title }) =>
      value.toLowerCase().includes(searchValue.toLowerCase()) ||
      title?.toLowerCase().includes(searchValue.toLowerCase())
  );

  const onSearchChange = (e) => {
    setSearchValue(e.target.value);
  };

  const clearSearch = () => setSearchValue("");

  return { searchValue, onSearchChange, clearSearch, filteredItems };
};

export default useSearch;

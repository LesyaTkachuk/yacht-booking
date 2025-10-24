import { useQuery } from "@tanstack/react-query";
import { StyledWrapper } from "./styled";
import { Typography, Stack } from "@mui/material";
import { getYachts } from "src/services/yachts";
import { YachtCard } from "src/components";

const HomePage = () => {
  const {
    isPending,
    error,
    data: yachts,
  } = useQuery({
    queryKey: ["yachts"],
    queryFn: getYachts,
  });

  // set filters to search params
  return (
    <>
      <StyledWrapper>
        <Stack width={"60%"}>
          <Typography variant="subtitle" color={"special.white"}>
            From sailing dreams to unforgettable escapes
          </Typography>
          <Typography variant="h1" color={"special.white"}>
            Choose the Perfect Yacht for Your Voyage
          </Typography>
        </Stack>
      </StyledWrapper>
      <Stack direction={"row"} gap={4} flexWrap={"wrap"} padding={4}>
        {yachts?.map((yacht) => (
          <YachtCard key={yacht.id} yachtDetails={yacht} />
        ))}
      </Stack>
    </>
  );
};

export default HomePage;

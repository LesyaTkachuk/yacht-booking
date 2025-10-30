import { useQuery, useQueryClient } from "@tanstack/react-query";
import { StyledWrapper } from "./styled";
import { Typography, Stack } from "@mui/material";
import { getYachts, getRecommendedYachts } from "src/services/yachts";
import { YachtCard } from "src/components";
import { useAuth } from "src/context/authContext";

const HomePage = () => {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const {
    isPending,
    error,
    data: yachts,
  } = useQuery({
    queryKey: ["yachts"],
    queryFn: getYachts,
  });

  const {
    isPending: isPendingRecommended,
    error: errorRecommended,
    data: recommendedYachts,
  } = useQuery({
    queryKey: ["recommendedYachts", user && user.id],
    queryFn: getRecommendedYachts,
    enabled: !!user?.id,
  });

  recommendedYachts?.forEach((yacht) => {
    queryClient.setQueryData(["yachts", yacht.id], yacht);
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
      {!!recommendedYachts?.length && (
        <Stack gap={4} padding={4}>
          <Typography variant="h5">Top recommended yachts for you</Typography>
          <Stack
            direction={"row"}
            gap={4}
            rowGap={4}
            flexWrap={"wrap"}
            justifyContent={"space-between"}
          >
            {/* {yachts?.map((yacht) => (
          <YachtCard key={yacht.id} yachtDetails={yacht} />
        ))} */}
            {recommendedYachts?.map((yacht) => (
              <YachtCard key={yacht.id} yachtDetails={yacht} />
            ))}
          </Stack>{" "}
        </Stack>
      )}
    </>
  );
};

export default HomePage;

using UnityEngine;
using UnityEngine.UI;

public class LiveEMGOverlay : MonoBehaviour
{
    public RectTransform barRectTransform;
    public RectTransform thresholdLineRectTransform;
    public Image barImage;
    public Image thresholdLineImage;

    private LiveEMGDisplay liveEMGDisplay;

    private void Start()
    {
        liveEMGDisplay = GetComponent<LiveEMGDisplay>();
    }

    private void Update()
    {
        if (liveEMGDisplay.displayLiveEMGValue)
        {
            float emgPercent = Mathf.Clamp(liveEMGDisplay.emgInteraction.LiveEMGValue(0) / 1023f, 0f, 1f);
            float threshPercent = Mathf.Clamp(liveEMGDisplay.emgInteraction.EMGThreshold / 1023f, 0f, 1f);

            // Update the size of the bar based on the percentage
            Vector2 barSize = barRectTransform.sizeDelta;
            barSize.x = 400 * emgPercent;
            barRectTransform.sizeDelta = barSize;

            // Update the position of the threshold line based on the percentage
            Vector2 threshPosition = thresholdLineRectTransform.anchoredPosition;
            threshPosition.x = 100 + 400 * threshPercent;
            thresholdLineRectTransform.anchoredPosition = threshPosition;
        }
    }
}